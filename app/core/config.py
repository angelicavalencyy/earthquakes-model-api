# for configuration settings using pydantic

from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import os

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Pydantic settings container for application configuration."""
    POSTGRES_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()

# Do not print or expose raw settings at import time. Log a redacted view at DEBUG when
# running in development so operators can verify values without leaking secrets.
try:
    raw = settings.model_dump()
    redacted = dict(raw)
    for key in ("POSTGRES_URL", "DATABASE_URL", "SECRET_KEY", "PASSWORD"):
        if key in redacted:
            redacted[key] = "REDACTED"
    if os.getenv("FASTAPI_DEV", "0").lower() in ("1", "true", "yes"):
        logger.debug("Loaded settings: %s", redacted)
except Exception:
    # Avoid raising during import; configuration issues will surface elsewhere.
    logger.debug("Could not dump settings for debug logging.")