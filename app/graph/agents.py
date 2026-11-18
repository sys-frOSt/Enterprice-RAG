
import logging
from collections.abc import Sequence

from langchain.agents import create_agent as create_langgraph_agent
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.llm.groq import create_groq
from app.mcp.tool_registry import load_tool_registry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an Enterprise AI Assistant.

Responsibilities:
- Answer company questions using RAG tools.
- Search and manage Jira issues using Jira tools.
- Combine document and Jira information when needed.

Rules:
- Never invent information.
- Use the appropriate tools.
- Ask for missing information.
- Require explicit approval before Jira write operations.
- Never claim an operation succeeded without tool confirmation.
"""


ALLOWED_JIRA_TOOLS = {
    "getJiraIssue",
    "searchJiraIssuesUsingJql",
    "createJiraIssue",
    "editJiraIssue",
    "transitionJiraIssue",
    "addOrEditJiraIssueComment",
}


async def create_agent(
    settings: Settings,
    rag_tools: Sequence[BaseTool],
    jira_email: str | None = None,
    jira_api_token: str | None = None,
):
    llm = create_groq(settings)

    mcp_registry = {}

    if jira_email and jira_api_token:
        logger.info("Loading Jira MCP tools...")

        mcp_registry = await load_tool_registry(
            jira_email,
            jira_api_token,
        )

        logger.info(
            "Loaded %d MCP tools",
            len(mcp_registry),
        )

    else:
        logger.warning("Jira credentials are missing.")

    jira_tools = [
        tool
        for name, tool in mcp_registry.items()
        if name in ALLOWED_JIRA_TOOLS
    ]

    all_tools = [
        *rag_tools,
        *jira_tools,
    ]

    logger.info(
        "Agent tools registered: %s",
        [tool.name for tool in all_tools],
    )

    if not all_tools:
        raise RuntimeError("No tools available for the Enterprise AI Agent")

    checkpointer = InMemorySaver()

    agent = create_langgraph_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    logger.info("Enterprise AI Agent initialized successfully")

    return agent
