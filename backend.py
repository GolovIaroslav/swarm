"""Backend abstraction.

Four modes:
  * lm_studio — expect a running LM Studio server at config.backend.url.
  * llama_cpp — spawn `llama-server` as a subprocess, wait for /v1/models,
    register cleanup so it dies with us.
  * custom   — assume someone else is running an OpenAI-compat endpoint.
  * api      — remote provider via LiteLLM (OpenRouter, NVIDIA NIM, Groq,
               OpenAI, Anthropic, Gemini, …). Model string carries the provider
               prefix; API key comes from an env var named in config.

Public surface:
  * Backend.start()    — bring the server up (no-op for lm_studio/custom/api).
  * Backend.stop()     — kill spawned subprocess if any.
  * Backend.ping()     — GET /v1/models, return list of model IDs.
  * Backend.llm(model_override=None) — build a configured crewai.LLM.
"""

from __future__ import annotations

import os
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
        """Bring the server up. For llama_cpp spawn the subprocess.
        For lm_studio / custom ping the server and prefer a LOADED model.
        For api verify the API key is present in the environment.
        """
        btype = self.cfg.backend.type

        if btype == "llama_cpp":
            self._spawn_llama_cpp()
        elif btype == "api":
            api = self.cfg.backend.api
            if not api.model:
                raise RuntimeError(
                    "backend.type='api' but backend.api.model is empty. "
                    "Set e.g. model='openrouter/anthropic/claude-3.5-sonnet'."
                )
            if api.api_key_env and not os.environ.get(api.api_key_env):
                raise RuntimeError(
                    f"Env var {api.api_key_env!r} is not set. "
                    "Export your provider API key before running."
                )
            self.model_id = api.model
        else:
            # lm_studio or custom — check the LM Studio native API first to
            # pick a LOADED model. If LM Studio isn't running OR no model is
            # loaded, fail fast with a clear message instead of letting the
            # first agent crash mid-thought.
            loaded = self._loaded_via_lm_studio()
            if loaded is not None:
                if not loaded:
                    raise RuntimeError(
                        "LM Studio is running but no model is loaded. "
                        "Open LM Studio, load a model (the 'Local Server' tab), "
                        "then run swarm again."
                    )
                self.model_id = loaded[0]
                return
            # fall back to standard /v1/models for non-LM-Studio servers
            try:
                models = self.ping()
            except Exception as e:
                raise RuntimeError(
                    f"Could not reach the LLM server at {self.cfg.backend.url}. "
                    f"Is it running? ({e})"
                )
            if not models:
                raise RuntimeError(
                    f"No models advertised at {self.cfg.backend.url}. "
                    "If this is LM Studio, load a model and try again."
                )
            self.model_id = models[0]

    def _loaded_via_lm_studio(self) -> Optional[list[str]]:
        """LM Studio's /api/v0/models exposes a `state` field per model.
        Returns the list of model ids whose state is 'loaded'.
        Returns None if the endpoint isn't available (i.e. not LM Studio)."""
        try:
            base = self.cfg.backend.url.rstrip("/")
            lm_url = base.replace("/v1", "/api/v0") + "/models"
            resp = requests.get(lm_url, timeout=3)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not isinstance(data, dict) or "data" not in data:
                return None
            return [
                m.get("id", "") for m in data["data"]
                if m.get("state") == "loaded" and m.get("id")
            ]
        except Exception:
            return None

    def stop(self) -> None:
        """Kill the spawned subprocess (if any). Safe to call multiple times."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def detect_context_window(self) -> Optional[int]:
        """Try to detect the loaded model's actual context length.

        LM Studio: GET /api/v0/models -> data[i].max_context_length or loaded_context_length
        llama.cpp: GET /props        -> default_generation_settings.n_ctx

        Returns None if no source matches. Caller should fall back to the
        configured cfg.execution.context_window.
        """
        if self.cfg.backend.type == "api":
            return None

        base = self.cfg.backend.url.rstrip("/")
        if self.cfg.backend.type == "llama_cpp":
            lc = self.cfg.backend.llama_cpp
            base = f"http://localhost:{lc.port}/v1"

        # LM Studio native API
        try:
            lm_url = base.replace("/v1", "/api/v0") + "/models"
            resp = requests.get(lm_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    ctx = m.get("loaded_context_length") or m.get("max_context_length")
                    if ctx:
                        return int(ctx)
        except Exception:
            pass

        # llama.cpp /props
        try:
            url = base.replace("/v1", "") + "/props"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                n_ctx = (
                    data.get("default_generation_settings", {}).get("n_ctx")
                    or data.get("n_ctx")
                )
                if n_ctx:
                    return int(n_ctx)
        except Exception:
            pass

        return None

    def ping(self) -> list[str]:
        """GET {base_url}/models -> list of available model IDs.

        For backend.type='api' returns the configured model as a single-item
        list (no models endpoint to hit — we trust the user).
        """
        if self.cfg.backend.type == "api":
            return [self.cfg.backend.api.model] if self.cfg.backend.api.model else []
        url = self.cfg.backend.url.rstrip("/") + "/models"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def llm(self, model_override: Optional[str] = None) -> LLM:
        """Return a configured crewai.LLM.

        For local backends, model id is prefixed with `openai/` so LiteLLM
        picks the native OpenAI SDK path (mandatory for local servers).
        For api backends, the user-provided model string already has the
        right provider prefix (openrouter/, nvidia_nim/, groq/, ...).

        `drop_params=True` is mandatory: local models choke on extra params.
        """
        if not self.model_id and not model_override:
            raise RuntimeError("model_id not set — call start() first")

        btype = self.cfg.backend.type

        if btype == "api":
            api = self.cfg.backend.api
            model = model_override or api.model
            api_key = os.environ.get(api.api_key_env, "") if api.api_key_env else ""
            kwargs = dict(
                model=model,
                api_key=api_key,
                max_tokens=self.cfg.execution.max_response_tokens,
                drop_params=True,
            )
            if api.base_url:
                kwargs["base_url"] = api.base_url
            return LLM(**kwargs)

        # local backends (lm_studio / llama_cpp / custom).
        # MANDATORY: always wrap with "openai/" prefix so LiteLLM uses the
        # OpenAI-compat path. Without it CrewAI auto-detects providers based
        # on the model name (e.g. "google/gemma-..." => native Gemini SDK,
        # which then errors out because the package isn't installed).
        base_url = self.cfg.backend.url
        if btype == "llama_cpp":
            lc = self.cfg.backend.llama_cpp
            base_url = f"http://localhost:{lc.port}/v1"

        bare = model_override or self.model_id
        # strip an already-present "openai/" prefix to avoid double-wrapping
        if bare.startswith("openai/"):
            bare = bare[len("openai/"):]
        model = f"openai/{bare}"

        return LLM(
            model=model,
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
