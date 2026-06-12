import os
import uvicorn

if __name__ == "__main__":
    # Safe production defaults: bind to all interfaces and no reload.
    reload_flag = os.getenv("FASTAPI_DEV", "0").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=reload_flag)
