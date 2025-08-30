#!/usr/bin/env python3
"""
Example demonstrating the new outputSchema feature in fastapi-mcp.

This example shows how the convert_openapi_to_mcp_tools function now automatically
extracts and includes outputSchema information from FastAPI response models.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
from fastapi.openapi.utils import get_openapi
import json


# Define response models
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


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None


# Create FastAPI app
app = FastAPI(title="Output Schema Example", description="Example showing outputSchema extraction", version="1.0.0")


@app.get(
    "/items",
    response_model=ItemList,
    operation_id="list_items",
    summary="List all items",
    description="Retrieve a paginated list of items",
)
async def list_items(page: int = 1, size: int = 10):
    """List all items with pagination."""
    return ItemList(
        items=[
            Item(id=1, name="Item 1", price=10.99, tags=["electronics"]),
            Item(id=2, name="Item 2", price=20.50, tags=["clothing"]),
        ],
        total=2,
        page=page,
        size=size,
    )


@app.get(
    "/items/{item_id}",
    response_model=Item,
    operation_id="get_item",
    summary="Get a specific item",
    description="Retrieve detailed information about a specific item",
)
async def get_item(item_id: int):
    """Get a specific item by ID."""
    return Item(id=item_id, name=f"Item {item_id}", price=15.99, tags=["electronics"])


@app.post(
    "/items",
    response_model=Item,
    operation_id="create_item",
    summary="Create a new item",
    description="Create a new item in the system",
)
async def create_item(item: Item):
    """Create a new item."""
    return item


@app.delete(
    "/items/{item_id}",
    operation_id="delete_item",
    summary="Delete an item",
    description="Delete an item from the system",
)
async def delete_item(item_id: int):
    """Delete an item by ID."""
    return {"message": f"Item {item_id} deleted successfully"}


def main():
    """Demonstrate the outputSchema feature."""
    print("=== Output Schema Example ===\n")

    # Generate OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    # Convert to MCP tools with output schema enabled (default)
    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema, include_output_schema=True)

    print(f"Generated {len(tools)} MCP tools:\n")

    for tool in tools:
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description[:100]}...")

        if tool.inputSchema:
            print(f"Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
        else:
            print("Input Schema: None")

        if tool.outputSchema:
            print(f"Output Schema: {json.dumps(tool.outputSchema, indent=2)}")
        else:
            print("Output Schema: None")

        print("-" * 50)

    print("\n=== Key Features ===")
    print("1. Tools with response models (list_items, get_item, create_item) have outputSchema")
    print("2. Tools without response models (delete_item) have outputSchema=None")
    print("3. outputSchema contains cleaned JSON schema from FastAPI response models")
    print("4. This helps LLMs understand the expected output structure")
    print("5. Use include_output_schema=False to disable outputSchema generation")

    # Demonstrate the include_output_schema parameter
    print("\n=== Demonstrating include_output_schema parameter ===")

    # Convert without output schema
    tools_without_schema, _ = convert_openapi_to_mcp_tools(openapi_schema, include_output_schema=False)

    print(f"\nTools without outputSchema: {len(tools_without_schema)}")
    for tool in tools_without_schema:
        print(f"  {tool.name}: outputSchema = {tool.outputSchema}")


if __name__ == "__main__":
    main()
