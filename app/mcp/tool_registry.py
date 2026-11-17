from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool

from app.mcp.jira_client import get_jira_tools

ToolLoader = Callable[[str, str], Awaitable[list[BaseTool]]]

TOOL_LOADERS: dict[str, ToolLoader] = {
    "jira": get_jira_tools,
}


async def load_tool_registry(email: str, api_token: str) -> dict[str, BaseTool]:
    registry: dict[str, BaseTool] = {}

    for loader in TOOL_LOADERS.values():
        for tool in await loader(email, api_token):
            if tool.name in registry:
                raise ValueError(f"Duplicate MCP tool name: {tool.name}")
            registry[tool.name] = tool

    return registry
