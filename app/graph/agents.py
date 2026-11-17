
import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.llm.groq import create_groq
from app.mcp.tool_registry import load_tool_registry

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an Enterprise AI Assistant.

Your responsibilities:
1. Answer company-related questions using RAG tools.
2. Search Jira issues using Jira MCP tools.
3. Create and update Jira issues when requested.
4. Combine RAG and Jira information when required.

Rules:
- Never invent information.
- Use the appropriate registered tools.
- For Jira operations, use the available Jira MCP tools.
- If a required Jira operation is unavailable, explain that clearly.
- Ask for missing information before executing an operation.
- Require explicit user approval before Jira write operations.
- After receiving approval, invoke the appropriate Jira write tool.
- Never claim an issue was created without a successful Jira tool result.
"""


async def create_agent(
    settings: Settings,
    rag_tools: Sequence[BaseTool],
    jira_email: str | None = None,
    jira_api_token: str | None = None,
):

    # Initialize LLM
    llm = create_groq(settings)

    # Load Jira MCP tools
    mcp_registry = {}

    if jira_email and jira_api_token:
        logger.info("Loading Jira MCP tools...")

        mcp_registry = await load_tool_registry(
            jira_email,
            jira_api_token,
        )

        logger.info(
            "Successfully loaded %d Jira MCP tools",
            len(mcp_registry),
        )

        for name, jira_tool in mcp_registry.items():
            logger.info(
                "Jira tool registered: %s | Description: %s",
                name,
                jira_tool.description[:200],
            )

    else:
        logger.warning(
            "Jira MCP tools were not loaded. "
            "Atlassian credentials are missing."
        )

    # Combine RAG and Jira tools
    all_tools = [
        *rag_tools,
        *mcp_registry.values(),
    ]

    logger.info(
        "Total agent tools registered: %s",
        [tool.name for tool in all_tools],
    )

    if not all_tools:
        raise RuntimeError("No tools available for the Enterprise AI Agent")

    # Initialize conversation memory
    checkpointer = InMemorySaver()

    # Create LangGraph agent
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    logger.info("Enterprise AI Agent initialized successfully")

    return agent
