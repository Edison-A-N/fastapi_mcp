import contextlib
import logging
import threading
from typing import Any, Dict
from anyio.abc import TaskStatus
import anyio

from functools import partial

from fastapi import FastAPI, Request
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import StreamableHTTPServerTransport

logger = logging.getLogger(__name__)


class StreamableHTTPSessionManagerLite:
    """
    Minimal session manager for streamable HTTP endpoints, without MCP server coupling.
    Only supports stateless mode and manages a task group for request handling.
    """

    def __init__(self):
        self._task_group = None
        self._run_lock = threading.Lock()
        self._has_started = False
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._stateless_transport: Dict[str, StreamableHTTPServerTransport] = {}

    @contextlib.asynccontextmanager
    async def run(self):
        """
        Initializes the task group for stateless streamable HTTP endpoints.
        Can only be called once per instance.
        """
        with self._run_lock:
            if self._has_started:
                raise RuntimeError(
                    "StreamableHTTPSessionManagerLite .run() can only be called once per instance. "
                    "Create a new instance if you need to run again."
                )
            self._has_started = True
        async with anyio.create_task_group() as tg:
            self._task_group = tg
            logger.info("StreamableHTTPSessionManagerLite started")
            try:
                yield
            finally:
                logger.info("StreamableHTTPSessionManagerLite shutting down")
                tg.cancel_scope.cancel()
                self._task_group = None
                self._registry.clear()
                self._stateless_transport.clear()

    def register(
        self,
        name: str,
        fastapi_app: FastAPI,
        mount_path: str,
        mcp_server: MCPServer[Any, Any],
        stateless: bool = True,
        json_response: bool = False,
        security_settings: Any = None,
        **kwargs,
    ):
        """
        Register a stateless streamable HTTP endpoint.
        Args:
            name: Unique name for the endpoint
            mount_path: Path to mount the endpoint
            mcp_server: The MCP server instance to use for this endpoint
            stateless: Only stateless mode is supported
            json_response: Whether to use JSON response mode
            security_settings: Optional security settings for the transport
            kwargs: Extra info for future use
        """
        if name in self._registry:
            raise ValueError(f"Endpoint with name '{name}' is already registered.")
        self._registry[name] = {
            "mount_path": mount_path,
            "mcp_server": mcp_server,
            "stateless": stateless,
            "json_response": json_response,
            "security_settings": security_settings,
            **kwargs,
        }
        logger.info(f"Registered stateless endpoint: {name} at {mount_path}")

        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"
        if mount_path.endswith("/"):
            mount_path = mount_path[:-1]

        fastapi_app.add_api_route(
            path=mount_path,
            endpoint=partial(self._handle_stateless_request, name),
            methods=["POST"],
            include_in_schema=False,
        )

    async def _start_stateless_server(self, name: str):
        if self._task_group is None:
            raise RuntimeError("Task group is not initialized. Use run() context manager.")
        if name not in self._registry:
            raise ValueError(f"No endpoint registered with name '{name}'")

        if self._stateless_transport.get(name):
            return

        entry = self._registry[name]
        mcp_server: MCPServer = entry["mcp_server"]
        json_response = entry.get("json_response", False)

        http_transport = StreamableHTTPServerTransport(
            mcp_session_id=None,
            is_json_response_enabled=json_response,
        )

        async def run_stateless_server(*, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED):
            async with http_transport.connect() as (reader, writer):
                task_status.started()
                await mcp_server.run(
                    reader,
                    writer,
                    mcp_server.create_initialization_options(),
                    stateless=True,
                )

        assert self._task_group is not None
        await self._task_group.start(run_stateless_server)

        self._stateless_transport[name] = http_transport

    async def _handle_stateless_request(self, name: str, request: Request):
        """
        Handle a stateless request for a registered endpoint.
        Args:
            scope, receive, send: ASGI parameters
        """
        await self._start_stateless_server(name=name)
        http_transport = self._stateless_transport.get(name)
        if not http_transport:
            raise ValueError("No Transport supported")

        await http_transport.handle_request(
            request.scope,
            request.receive,
            request._send,
        )

    @property
    def registry(self):
        return self._registry
