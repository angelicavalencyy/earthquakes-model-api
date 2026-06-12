import os
import logging
import uvicorn

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Safe production defaults: bind to all interfaces and no reload.
    reload_flag = os.getenv("FASTAPI_DEV", "0").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # When running in dev mode prefer binding to localhost for safety and
    # show a user-friendly localhost URL in logs. If HOST explicitly set,
    # respect it for binding but still display localhost if bound to 0.0.0.0.
    bind_host = host
    if reload_flag and host in ("0.0.0.0", "::"):
        bind_host = "127.0.0.1"

    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    logger.info("Starting server at http://%s:%s (reload=%s)", display_host, port, reload_flag)

    uvicorn.run("app.main:app", host=bind_host, port=port, reload=reload_flag)
