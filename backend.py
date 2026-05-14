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
from dataclasses import dataclass
from typing import Optional

from config import Config


@dataclass
class Backend:
    cfg: Config
    proc: Optional[subprocess.Popen] = None
    model_id: str = ""

    def start(self) -> None:
        """Bring the server up. For llama_cpp: spawn subprocess and wait for
        /v1/models. For lm_studio/custom: just verify the server answers."""
        raise NotImplementedError("session 2")

    def stop(self) -> None:
        """Kill the spawned subprocess (if any). Safe to call multiple times."""
        raise NotImplementedError("session 2")

    def ping(self) -> list[str]:
        """GET {base_url}/models -> list of available model IDs."""
        raise NotImplementedError("session 2")

    def llm(self):
        """Return a configured crewai.LLM bound to the active model.

        IMPORTANT: model id must be `f"openai/{self.model_id}"` — the
        openai/ prefix is what makes CrewAI pick the native OpenAI SDK path.
        Also pass `drop_params=True` (local models choke on unknown params).
        """
        raise NotImplementedError("session 2")


def detect_model_id(base_url: str) -> str:
    """Hit /v1/models and pick the first available model id."""
    raise NotImplementedError("session 2")
