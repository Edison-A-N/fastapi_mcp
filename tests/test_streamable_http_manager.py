import pytest
import json
import httpx
import uvicorn
import socket
import time
import multiprocessing
from fastapi import FastAPI
from fastapi_mcp.transport.streamable_http_manager import StreamableHTTPSessionManagerLite

from fastapi_mcp import FastApiMCP
from .fixtures.simple_app import make_simple_fastapi_app

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from mcp.types import InitializeResult, EmptyResult


def process_result(resp):
    """
    处理 httpx.Response，返回 JSON-RPC 格式的 dict。
    支持 application/json 和 text/event-stream。
    """
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        return resp.json()
    elif "text/event-stream" in content_type:
        # 只取第一个 data: 行
        lines = resp.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise ValueError("No data: line found in SSE response")
    else:
        raise ValueError(f"Unsupported content-type: {content_type}")


# Minimal mock MCPServer
class MockMCPServer:
    def __init__(self, response_data):
        self.response_data = response_data

    async def run(self, reader, writer, init_options, stateless=False):
        # Simulate writing a response
        await writer.send(self.response_data)

    def create_initialization_options(self):
        return {}


HOST = "127.0.0.1"
SERVER_NAME = "Test MCP Server"


def get_free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def run_server(port, response_data):
    manager = StreamableHTTPSessionManagerLite()

    async def lifespan(app: FastAPI):
        async with manager.run():
            yield

    app = make_simple_fastapi_app(lifespan)

    mcp_server = FastApiMCP(
        app,
        name=SERVER_NAME,
        description="Test Description",
    )
    # Register endpoint before starting server
    manager.register(
        name="mcp_test",
        fastapi_app=app,
        mount_path="/mcp",
        mcp_server=mcp_server.server,
    )

    uvicorn.run(app, host=HOST, port=port, log_level="error")


@pytest.fixture(scope="session")
def server_port():
    return get_free_port()


@pytest.fixture(scope="session")
def server_url(server_port):
    return f"http://{HOST}:{server_port}"


@pytest.fixture(scope="session")
def server(server_port):
    proc = multiprocessing.Process(target=run_server, args=(server_port, "hello"), daemon=True)
    proc.start()
    # Wait for server to start
    for _ in range(30):
        try:
            with socket.create_connection((HOST, server_port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")
    yield
    proc.terminate()
    proc.join(timeout=2)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)


@pytest.fixture
async def http_client(server, server_url):
    async with httpx.AsyncClient(base_url=server_url) as client:
        yield client


@pytest.mark.anyio
async def test_stateless_endpoint(http_client):
    resp = await http_client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json={"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "test-ping"},
    )
    assert resp.status_code == 200, resp.text

    data = process_result(resp)
    assert data["id"] == "test-ping", data
    # assert data is None, data
    assert data["result"] == {}, data


@pytest.mark.anyio
async def test_streamable_http_basic_connection(server: None, server_url: str) -> None:
    async with streamablehttp_client(server_url + "/mcp") as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            # Test initialization
            result = await session.initialize()
            assert isinstance(result, InitializeResult)
            assert result.serverInfo.name == SERVER_NAME

            # Test ping
            ping_result = await session.send_ping()
            assert isinstance(ping_result, EmptyResult)


@pytest.mark.anyio
async def test_streamable_http_tool_call(server_url):
    async with streamablehttp_client(server_url + "/mcp") as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_list_result = await session.list_tools()
            assert len(tools_list_result.tools) > 0, tools_list_result.tools

            tool_call_result = await session.call_tool("get_item", {"item_id": 1})
            assert not tool_call_result.isError, tool_call_result.content
            assert tool_call_result.content is not None, tool_call_result.content
            assert len(tool_call_result.content) > 0, tool_call_result.content
