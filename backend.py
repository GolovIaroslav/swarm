"""Backend abstraction over the OpenAI-compatible local server.

Three modes:
  * lm_studio — expect a running LM Studio server at config.backend.url.
  * llama_cpp — spawn `llama-server` as a subprocess, wait for /v1/models,
    register cleanup so it dies with us.
  * custom   — assume someone else is running an OpenAI-compat endpoint.

Public surface:
  * Backend.start()    — bring the server up (no-op for lm_studio/custom).
  * Backend.stop()     — kill spawned subprocess if any.
  * Backend.ping()     — GET /v1/models, return list of model IDs.
  * Backend.llm()      — build a configured crewai.LLM for the active model.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from crewai import LLM

from config import Config


@dataclass
class Backend:
    cfg: Config
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    model_id: str = ""

    def start(self) -> None:
        """Bring the server up. For llama_cpp: spawn subprocess and wait for
        /v1/models. For lm_studio/custom: just verify the server answers."""
        btype = self.cfg.backend.type

        if btype == "llama_cpp":
            self._spawn_llama_cpp()
        else:
            # lm_studio or custom — just ping to verify
            models = self.ping()
            if not models:
                raise RuntimeError(
                    f"No models found at {self.cfg.backend.url}. "
                    "Is LM Studio running with a loaded model?"
                )
            self.model_id = models[0]

    def stop(self) -> None:
        """Kill the spawned subprocess (if any). Safe to call multiple times."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def ping(self) -> list[str]:
        """GET {base_url}/models -> list of available model IDs."""
        url = self.cfg.backend.url.rstrip("/") + "/models"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def llm(self) -> LLM:
        """Return a configured crewai.LLM bound to the active model.

        IMPORTANT: model id must be `f"openai/{self.model_id}"` — the
        openai/ prefix is what makes CrewAI pick the native OpenAI SDK path.
        Also pass `drop_params=True` (local models choke on unknown params).
        """
        if not self.model_id:
            raise RuntimeError("model_id not set — call start() first")

        base_url = self.cfg.backend.url
        if self.cfg.backend.type == "llama_cpp":
            lc = self.cfg.backend.llama_cpp
            base_url = f"http://localhost:{lc.port}/v1"

        return LLM(
            model=f"openai/{self.model_id}",
            base_url=base_url,
            api_key="lm-studio",
            max_tokens=self.cfg.execution.max_response_tokens,
            drop_params=True,
        )

    # ------------------------------------------------------------------
    def _spawn_llama_cpp(self) -> None:
        lc = self.cfg.backend.llama_cpp
        cmd = [
            lc.binary,
            "--model", lc.model,
            "--ctx-size", str(lc.ctx),
            "--n-gpu-layers", str(lc.ngl),
            "--port", str(lc.port),
            "--host", "127.0.0.1",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # wait up to 60s for the server to be ready
        base_url = f"http://localhost:{lc.port}/v1"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                resp = requests.get(base_url + "/models", timeout=2)
                if resp.status_code == 200:
                    models = [m["id"] for m in resp.json().get("data", [])]
                    self.model_id = models[0] if models else "local"
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError("llama-server did not come up within 60 seconds")


def detect_model_id(base_url: str) -> str:
    """Hit /v1/models and pick the first available model id."""
    url = base_url.rstrip("/") + "/models"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models available at {base_url}")
    return data[0]["id"]
