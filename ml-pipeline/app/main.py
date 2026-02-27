from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import configure_logging

settings = get_settings()
configure_logging()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.include_router(router)
frontend_path = Path(__file__).resolve().parent / "static" / "index.html"


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(frontend_path)
