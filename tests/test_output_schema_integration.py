import pytest
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi_mcp import FastApiMCP


# Define response models for testing
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    tags: List[str] = []


class ItemList(BaseModel):
    items: List[Item]
    total: int
    page: int
    size: int


def create_test_app() -> FastAPI:
    """Create a test FastAPI app with various endpoints."""
    app = FastAPI(title="Test App", description="Test app for output schema integration")

    items = [
        Item(id=1, name="Item 1", price=10.99, tags=["electronics"], description="First item"),
        Item(id=2, name="Item 2", price=20.50, tags=["clothing"]),
    ]

    @app.get("/items", response_model=ItemList, operation_id="list_items")
    async def list_items(page: int = 1, size: int = 10):
        """List all items with pagination."""
        return ItemList(
            items=items,
            total=len(items),
            page=page,
            size=size,
        )

    @app.get("/items/{item_id}", response_model=Item, operation_id="get_item")
    async def get_item(item_id: int):
        """Get a specific item by ID."""
        item = next((item for item in items if item.id == item_id), None)
        if item is None:
            raise ValueError(f"Item {item_id} not found")
        return item

    @app.post("/items", response_model=Item, operation_id="create_item")
    async def create_item(item: Item):
        """Create a new item."""
        return item

    @app.delete("/items/{item_id}", operation_id="delete_item")
    async def delete_item(item_id: int):
        """Delete an item by ID."""
        return {"message": f"Item {item_id} deleted successfully"}

    @app.get("/simple", operation_id="simple_endpoint")
    async def simple_endpoint():
        """Simple endpoint without response model."""
        return {"status": "ok", "message": "simple response"}

    return app


@pytest.fixture
def test_app():
    return create_test_app()


@pytest.fixture
def mcp_server_with_output_schema(test_app):
    """Create MCP server with output schema enabled."""
    return FastApiMCP(
        test_app,
        include_output_schema=True,
        prefer_structured_content=True,
    )


@pytest.fixture
def mcp_server_without_output_schema(test_app):
    """Create MCP server without output schema."""
    return FastApiMCP(
        test_app,
        include_output_schema=False,
        prefer_structured_content=False,
    )


def test_tool_creation_with_output_schema(mcp_server_with_output_schema):
    """Test that tools are created correctly with output schema."""
    mcp_server = mcp_server_with_output_schema

    # Check that tools have outputSchema when they have response models
    tools_by_name = {tool.name: tool for tool in mcp_server.tools}

    # Tools with response models should have outputSchema
    assert tools_by_name["list_items"].outputSchema is not None
    assert tools_by_name["get_item"].outputSchema is not None
    assert tools_by_name["create_item"].outputSchema is not None

    # Tools without response models should not have outputSchema
    assert tools_by_name["delete_item"].outputSchema is None
    assert tools_by_name["simple_endpoint"].outputSchema is None

    # Verify outputSchema structure for list_items
    list_items_schema = tools_by_name["list_items"].outputSchema
    assert list_items_schema["type"] == "object"
    assert "items" in list_items_schema["properties"]
    assert "total" in list_items_schema["properties"]
    assert "page" in list_items_schema["properties"]
    assert "size" in list_items_schema["properties"]


def test_tool_creation_without_output_schema(mcp_server_without_output_schema):
    """Test that tools are created correctly without output schema."""
    mcp_server = mcp_server_without_output_schema

    for tool in mcp_server.tools:
        assert tool.outputSchema is None, f"Tool {tool.name} should not have outputSchema"


@pytest.mark.asyncio
async def test_tool_execution_with_structured_content(mcp_server_with_output_schema):
    """Test that tool execution returns structured content when prefer_structured_content=True."""
    mcp_server = mcp_server_with_output_schema

    result = await mcp_server._execute_api_tool(
        client=mcp_server._http_client,
        tool_name="list_items",
        arguments={"page": 1, "size": 5},
        operation_map=mcp_server.operation_map,
    )

    assert isinstance(result, dict)
    assert "items" in result
    assert "total" in result
    assert "page" in result
    assert "size" in result

    result = await mcp_server._execute_api_tool(
        client=mcp_server._http_client,
        tool_name="get_item",
        arguments={"item_id": 1},
        operation_map=mcp_server.operation_map,
    )

    # Should return structured data
    assert isinstance(result, dict)
    assert "id" in result
    assert "name" in result
    assert "price" in result


@pytest.mark.asyncio
async def test_tool_execution_without_structured_content(mcp_server_without_output_schema):
    """Test that tool execution returns text content when prefer_structured_content=False."""
    mcp_server = mcp_server_without_output_schema

    # Test the _execute_api_tool method directly
    # This simulates what happens when an MCP client calls a tool

    # Test list_items tool
    result = await mcp_server._execute_api_tool(
        client=mcp_server._http_client,
        tool_name="list_items",
        arguments={"page": 1, "size": 5},
        operation_map=mcp_server.operation_map,
    )

    # When prefer_structured_content=False, the result should be a list of TextContent
    assert isinstance(result, list)
    assert len(result) == 1
    assert hasattr(result[0], "type")
    assert result[0].type == "text"

    # Parse the text content to verify it contains the expected data

    text_content = result[0].text
    data = json.loads(text_content)
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data


def test_default_behavior():
    """Test the default behavior of the parameters."""
    app = create_test_app()

    # Default behavior (both should be False)
    mcp = FastApiMCP(app)
    assert mcp._include_output_schema is False
    assert mcp._prefer_structured_content is False

    # Verify no tools have outputSchema by default
    for tool in mcp.tools:
        assert tool.outputSchema is None


@pytest.mark.asyncio
async def test_real_mcp_client_with_output_schema():
    """Test real MCP client interaction with output schema enabled."""
    from mcp.shared.memory import create_connected_server_and_client_session
    import mcp.types as types

    app = create_test_app()

    # Create MCP server with output schema enabled
    mcp_server = FastApiMCP(
        app,
        include_output_schema=True,
        prefer_structured_content=True,
    )
    mcp_server.mount_http()

    # Test real MCP client interaction
    async with create_connected_server_and_client_session(mcp_server.server) as client_session:
        # Test 1: List tools and verify outputSchema
        tools_result = await client_session.list_tools()
        assert len(tools_result.tools) > 0

        # Find tools with outputSchema
        tools_with_schema = [tool for tool in tools_result.tools if tool.outputSchema is not None]
        tools_without_schema = [tool for tool in tools_result.tools if tool.outputSchema is None]

        # Tools with response models should have outputSchema
        assert len(tools_with_schema) >= 3  # list_items, get_item, create_item
        assert len(tools_without_schema) >= 2  # delete_item, simple_endpoint

        # Verify specific tools have outputSchema
        list_items_tool = next(tool for tool in tools_result.tools if tool.name == "list_items")
        assert list_items_tool.outputSchema is not None
        assert list_items_tool.outputSchema["type"] == "object"
        assert "items" in list_items_tool.outputSchema["properties"]

        # Test 2: Call tool and verify structured response
        response = await client_session.call_tool("list_items", {"page": 1, "size": 5})

        assert not response.isError
        assert len(response.content) > 0

        # When prefer_structured_content=True and include_output_schema=True,
        # the response should have structuredContent
        assert response.structuredContent is not None

        # Verify structuredContent has the correct structure
        structured_data = response.structuredContent
        assert "items" in structured_data
        assert "total" in structured_data
        assert "page" in structured_data
        assert "size" in structured_data
        assert isinstance(structured_data["items"], list)

        # Also verify the content field is present (for backward compatibility)
        text_content = next(c for c in response.content if isinstance(c, types.TextContent))
        result = json.loads(text_content.text)

        # Verify the response structure matches the outputSchema
        assert "items" in result
        assert "total" in result
        assert "page" in result
        assert "size" in result
        assert isinstance(result["items"], list)

        # Test 3: Call tool without outputSchema
        response = await client_session.call_tool("delete_item", {"item_id": 999})

        assert not response.isError
        assert len(response.content) > 0

        # When prefer_structured_content=True, structuredContent should be present
        # (even for tools without outputSchema)
        assert response.structuredContent is not None

        text_content = next(c for c in response.content if isinstance(c, types.TextContent))
        result = json.loads(text_content.text)

        # Should still return structured data (dict) even without outputSchema
        assert "message" in result
        assert "Item 999 deleted successfully" in result["message"]


@pytest.mark.asyncio
async def test_real_mcp_client_without_structured_content():
    """Test real MCP client interaction without structured content."""
    from mcp.shared.memory import create_connected_server_and_client_session
    import mcp.types as types

    app = create_test_app()

    # Create MCP server without structured content
    mcp_server = FastApiMCP(
        app,
        include_output_schema=False,
        prefer_structured_content=False,
    )
    mcp_server.mount_http()

    # Test real MCP client interaction
    async with create_connected_server_and_client_session(mcp_server.server) as client_session:
        # Test: Call tool and verify no structured content
        response = await client_session.call_tool("delete_item", {"item_id": 999})

        assert not response.isError
        assert len(response.content) > 0

        # When prefer_structured_content=False, structuredContent should be None
        assert response.structuredContent is None

        text_content = next(c for c in response.content if isinstance(c, types.TextContent))
        result = json.loads(text_content.text)

        # Should return structured data in content field
        assert "message" in result
        assert "Item 999 deleted successfully" in result["message"]
