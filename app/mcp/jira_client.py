import base64

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_jira_tools(email: str, api_token: str) -> list[BaseTool]:
    if not email or not api_token:
        raise ValueError("Atlassian credentials are missing")

    credentials = f"{email}:{api_token}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    client = MultiServerMCPClient(
        {
            "jira": {
                "transport": "streamable_http",
                "url": "https://mcp.atlassian.com/v2/mcp",
                "headers": {"Authorization": f"Basic {encoded_credentials}"},
            }
        }
    )

    return await client.get_tools()
