"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import dataset, deferred, health, live, model
from .config import cors_origins

app = FastAPI(
    title="Godavari River Basin Flood Forecasting API",
    version="0.1.0",
    description="Research and demonstration API; not an emergency warning service.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(health.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(deferred.router, prefix="/api")