"""Minimal HTTP health server for worker processes.

Exposes GET /ready on a configurable port so Kubernetes / Fly.io
readiness probes know when the worker is connected and consuming.

Usage (inside a worker's main())::

    from graphrag.workers.health_server import HealthServer
    health = HealthServer(port=8080)
    await health.start()
    # ... start consumer ...
    health.set_ready()

The server runs as a background asyncio task and does not block the
consumer loop.  On SIGTERM the worker's main() cancels the consumer
task; health.stop() is called in the finally block.
"""

from __future__ import annotations

import asyncio
import json
from aiohttp import web

import structlog

log = structlog.get_logger(__name__)


class HealthServer:
    """
    Lightweight aiohttp server exposing:
      GET /ready   → 200 {"status": "ready"} | 503 {"status": "starting"}
      GET /live    → 200 {"status": "alive"} (always — proves process is running)
    """

    def __init__(self, port: int = 8080, worker_name: str = "worker"):
        self._port = port
        self._worker_name = worker_name
        self._ready = False
        self._runner: web.AppRunner | None = None

    def set_ready(self, ready: bool = True) -> None:
        self._ready = ready
        log.info("worker.health_ready", worker=self._worker_name, ready=ready)

    async def _handle_ready(self, request: web.Request) -> web.Response:
        status = "ready" if self._ready else "starting"
        code   = 200    if self._ready else 503
        return web.Response(
            status=code,
            content_type="application/json",
            text=json.dumps({"status": status, "worker": self._worker_name}),
        )

    async def _handle_live(self, request: web.Request) -> web.Response:
        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps({"status": "alive", "worker": self._worker_name}),
        )

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/ready", self._handle_ready)
        app.router.add_get("/live",  self._handle_live)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host="0.0.0.0", port=self._port)
        await site.start()
        log.info("worker.health_server_started", port=self._port, worker=self._worker_name)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            log.info("worker.health_server_stopped", worker=self._worker_name)
