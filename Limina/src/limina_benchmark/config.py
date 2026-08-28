"""Local-development configuration with environment variables taking precedence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    limina_api_key: str | None
    limina_enabled: bool
    limina_profile: str
    limina_export_html: bool
    recruiter_base_url: str | None
    recruiter_internal_api_key: str | None
    request_timeout_s: float

    @classmethod
    def from_environment(cls) -> "Settings":
        # Local-only convenience. Existing environment values always win, and this
        # function never logs or returns a secret beyond the in-process settings.
        # Restrict discovery to this harness's working directory. Searching parent
        # folders can accidentally consume another project's credentials.
        environment = dict(os.environ)
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        return cls(
            limina_api_key=os.getenv("LIMINA_API_KEY") or None,
            limina_enabled=os.getenv("LIMINA_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            limina_profile=os.getenv("LIMINA_PROFILE", "standard").strip(),
            limina_export_html=os.getenv("LIMINA_EXPORT_HTML", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            recruiter_base_url=os.getenv("RECRUITER_BASE_URL") or None,
            # Recruiter Agent itself names this credential INTERNAL_API_KEY. The
            # benchmark-specific name takes precedence when both are present.
            recruiter_internal_api_key=(
                environment.get("RECRUITER_INTERNAL_API_KEY")
                or environment.get("INTERNAL_API_KEY")
                or os.getenv("RECRUITER_INTERNAL_API_KEY")
                or os.getenv("INTERNAL_API_KEY")
                or None
            ),
            request_timeout_s=float(os.getenv("EVAL_REQUEST_TIMEOUT_S", "60")),
        )


def default_results_dir() -> Path:
    return Path("results")
