## ⚠️ Project Status Notice

**This project is currently archived. Future deletion is possible.**

After careful consideration and community feedback, I've determined that converting RESTful APIs to MCP tools is not the recommended approach for production use. This project was primarily experimental in nature, and I've found that the current maintenance level is insufficient for continued active development.

For those interested in experimenting with MCP integrations, I recommend using [FastMCP](https://gofastmcp.com/integrations/fastapi) instead, which provides a more robust foundation for such experiments.

**Further reading on this topic:**
- [Stop Converting Your REST APIs to MCP](https://www.jlowin.dev/blog/stop-converting-rest-apis-to-mcp) by Jason Lowin
- [Stop Converting REST API to MCP](https://edison-a-n.github.io/2025/09/29/stop-converting-rest-api-to-mcp/) by Edison
- [FastMCP FastAPI Integration Guide](https://gofastmcp.com/integrations/fastapi)
- [FastMCP OpenAPI Integration Guide](https://gofastmcp.com/integrations/openapi)

---

## 🚀 Enhanced Features (feature-test branch)

**Built by [@Edison](https://github.com/Edison-A-N)**

- **Stateless Mode**: Force stateless transport by ignoring mcp-session-id headers ([PR #213](https://github.com/tadata-org/fastapi_mcp/pull/213) 🔄)
- **Skip Deprecated APIs**: Automatically exclude deprecated operations (enabled by default) ([PR #180](https://github.com/tadata-org/fastapi_mcp/pull/180) 🔄)
- **Structured Content Response**: Return structured JSON responses instead of text content ([PR #230](https://github.com/tadata-org/fastapi_mcp/pull/230) 🔄)
- **Response Info Control**: Fine-grained control over response information in tool descriptions ([PR #215](https://github.com/tadata-org/fastapi_mcp/pull/215) 🔄)
- **Output Schema Support**: Include outputSchema in MCP tools for better LLM understanding ([feature/output-schema-support](https://github.com/Edison-A-N/fastapi_mcp/tree/feature/output-schema-support) ⏳)

These features are available in the `feature-test` branch.

---

<p align="center"><a href="https://github.com/tadata-org/fastapi_mcp"><img src="https://github.com/user-attachments/assets/7e44e98b-a0ba-4aff-a68a-4ffee3a6189c" alt="fastapi-to-mcp" height=100/></a></p>

<div align="center">
  <span style="font-size: 0.85em; font-weight: normal;">Built by <a href="https://tadata.com">Tadata</a></span>
</div>

<h1 align="center">
  FastAPI-MCP
</h1>

<div align="center">
<a href="https://trendshift.io/repositories/14064" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14064" alt="tadata-org%2Ffastapi_mcp | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<p align="center">Expose your FastAPI endpoints as Model Context Protocol (MCP) tools, with Auth!</p>
<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/fastapi-mcp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/fastapi-mcp/)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastapi-mcp.svg)](https://pypi.org/project/fastapi-mcp/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white)](#)
[![CI](https://github.com/tadata-org/fastapi_mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/tadata-org/fastapi_mcp/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/tadata-org/fastapi_mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/tadata-org/fastapi_mcp)

</div>


<p align="center"><a href="https://github.com/tadata-org/fastapi_mcp"><img src="https://github.com/user-attachments/assets/b205adc6-28c0-4e3c-a68b-9c1a80eb7d0c" alt="fastapi-mcp-usage" height="400"/></a></p>


## Features

- **Authentication** built in, using your existing FastAPI dependencies!

- **FastAPI-native:** Not just another OpenAPI -> MCP converter

- **Zero/Minimal configuration** required - just point it at your FastAPI app and it works

- **Preserving schemas** of your request models and response models

- **Preserve documentation** of all your endpoints, just as it is in Swagger

- **Flexible deployment** - Mount your MCP server to the same app, or deploy separately

- **ASGI transport** - Uses FastAPI's ASGI interface directly for efficient communication


## Hosted Solution

If you prefer a managed hosted solution check out [tadata.com](https://tadata.com).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/), a fast Python package installer:

```bash
uv add fastapi-mcp
```

Alternatively, you can install with pip:

```bash
pip install fastapi-mcp
```

## Basic Usage

The simplest way to use FastAPI-MCP is to add an MCP server directly to your FastAPI application:

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

mcp = FastApiMCP(app)

# Mount the MCP server directly to your FastAPI app
mcp.mount()
```

That's it! Your auto-generated MCP server is now available at `https://app.base.url/mcp`.

## Documentation, Examples and Advanced Usage

FastAPI-MCP provides [comprehensive documentation](https://fastapi-mcp.tadata.com/). Additionaly, check out the [examples directory](examples) for code samples demonstrating these features in action.

## FastAPI-first Approach

FastAPI-MCP is designed as a native extension of FastAPI, not just a converter that generates MCP tools from your API. This approach offers several key advantages:

- **Native dependencies**: Secure your MCP endpoints using familiar FastAPI `Depends()` for authentication and authorization

- **ASGI transport**: Communicates directly with your FastAPI app using its ASGI interface, eliminating the need for HTTP calls from the MCP to your API

- **Unified infrastructure**: Your FastAPI app doesn't need to run separately from the MCP server (though [separate deployment](https://fastapi-mcp.tadata.com/advanced/deploy#deploying-separately-from-original-fastapi-app) is also supported)

This design philosophy ensures minimum friction when adding MCP capabilities to your existing FastAPI services.


## Development and Contributing

Thank you for considering contributing to FastAPI-MCP! We encourage the community to post Issues and create Pull Requests.

Before you get started, please see our [Contribution Guide](CONTRIBUTING.md).

## Community

Join [MCParty Slack community](https://join.slack.com/t/themcparty/shared_invite/zt-30yxr1zdi-2FG~XjBA0xIgYSYuKe7~Xg) to connect with other MCP enthusiasts, ask questions, and share your experiences with FastAPI-MCP.

## Requirements

- Python 3.10+ (Recommended 3.12)
- uv

## License

MIT License. Copyright (c) 2025 Tadata Inc.
