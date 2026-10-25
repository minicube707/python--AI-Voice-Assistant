from nlp.command import Command
from nlp.intent import AI_AGENT_TRIGGERS

from langchain.agents import create_agent
from ai_agent.tools import search_tool, wiki_tool, get_weather
from ai_agent.sandbox import run_python_in_sandbox
from ai_agent.ensure_program_running import ensure_ollama_running

import logging
import json
from pprint import pformat
from datetime import datetime

logger = logging.getLogger("ai_agent")

class AIAgent:

    def __init__(self, backend: str, model: str):
        
        tools = [
            search_tool,
            wiki_tool,
            get_weather,
            run_python_in_sandbox
        ]

        self.backend=backend
        self.agent = create_agent(
            model=set_model(backend, model),
            tools=tools,
            system_prompt = """
                Tu es un assistant vocal personnel.

                Réponds de manière naturelle, claire, concise et adaptée à une conversation orale.

                Règles :
                - Texte brut uniquement, sans Markdown.
                - Fais des réponses courtes et directes.
                - Ne répète pas inutilement la question.
                - Pour une question simple, réponds en une ou deux phrases.
                - Si une information importante manque, pose une question courte.
                - Utilise les outils disponibles lorsqu'ils permettent d'obtenir une information actuelle ou d'effectuer une action.
                - Ne devine jamais une information que tu peux vérifier avec un outil.
                - Ne mentionne pas les outils ni leur fonctionnement à l'utilisateur.
                - Pour les actions importantes ou irréversibles, demande confirmation.
                - La réponse sera lue à voix haute : privilégie des phrases naturelles et faciles à écouter.
            """
        )


    def call_agent(self, query: str):

        ensure_ollama_running()

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )
        
        self.debug_ai_agent(result)

        response = result["messages"][-1]
        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return str(content)


    def debug_ai_agent(self, result):
        """Log the agent's execution in a readable format."""

        messages = result.get("messages", [])

        logger.debug("=" * 80)
        logger.debug("🧠 DEBUG AI AGENT")
        logger.debug("📅 %s", datetime.now().strftime("%H:%M:%S"))
        logger.debug("📨 Number of messages: %d", len(messages))
        logger.debug("=" * 80)

        for i, message in enumerate(messages, start=1):
            msg_type = message.__class__.__name__
            content = getattr(message, "content", None)

            logger.debug("")
            logger.debug("─" * 80)
            logger.debug("📨 MESSAGE %d/%d | %s", i, len(messages), msg_type)
            logger.debug("─" * 80)

            # Useful metadata
            if getattr(message, "name", None):
                logger.debug("Name       : %s", message.name)

            if getattr(message, "id", None):
                logger.debug("Message ID : %s", message.id)

            usage_metadata = getattr(message, "usage_metadata", None)
            if usage_metadata:
                logger.debug("📊 TOKEN USAGE:\n%s", pformat(
                    usage_metadata, sort_dicts=False
                ))

            # Tool calls decided by the model
            tool_calls = getattr(message, "tool_calls", []) or []

            if tool_calls:
                logger.debug("🤖 TOOL CALLS")

                for j, tool in enumerate(tool_calls, start=1):
                    logger.debug(
                        "\n  [%d] 🔧 %s",
                        j,
                        tool.get("name", "unknown"),
                    )
                    logger.debug("  ID : %s", tool.get("id", "N/A"))
                    logger.debug("  Arguments :")

                    args = tool.get("args", {})

                    try:
                        args_str = json.dumps(
                            args,
                            indent=4,
                            ensure_ascii=False,
                            default=str,
                        )
                    except (TypeError, ValueError):
                        args_str = pformat(args, sort_dicts=False)

                    logger.debug("%s", args_str)

            # Invalid calls
            invalid_calls = (
                getattr(message, "invalid_tool_calls", []) or []
            )

            if invalid_calls:
                logger.warning(
                    "⚠️ INVALID TOOL CALLS:\n%s",
                    pformat(invalid_calls, sort_dicts=False),
                )

            # Result from a tool
            if msg_type == "ToolMessage":
                logger.debug("🔧 TOOL RESULT")
                logger.debug("Tool       : %s", getattr(message, "name", "N/A"))
                logger.debug(
                    "Tool call  : %s",
                    getattr(message, "tool_call_id", "N/A"),
                )
                logger.debug("Output :")
                self._debug_content(content)

            # Message content
            elif content:
                logger.debug("💬 CONTENT")
                self._debug_content(content)

            # Additional metadata
            response_metadata = getattr(message, "response_metadata", None)

            if response_metadata:
                logger.debug(
                    "📋 RESPONSE METADATA\n%s",
                    pformat(response_metadata, sort_dicts=False),
                )

        logger.debug("=" * 80)
        logger.debug("✅ DEBUG ENDED")
        logger.debug("=" * 80)


    def _debug_content(self, content):
        """Displays the content of a message in a clean format."""

        if isinstance(content, str):
            logger.debug(content)
            return

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")

                    if block_type == "text":
                        logger.debug(block.get("text", ""))

                    elif block_type == "image":
                        logger.debug("🖼️ [Image block]")

                    elif block_type in ("tool_use", "tool_call"):
                        logger.debug(f"🔧 [Tool call block: {block_type}]")
                        logger.debug(pformat(block, sort_dicts=False))

                    else:
                        logger.debug(f"📦 [Bloc : {block_type}]")
                        logger.debug(pformat(block, sort_dicts=False))
                else:
                    logger.debug(pformat(block, sort_dicts=False))

            return

        logger.debug(pformat(content, sort_dicts=False))


def set_model(backend: str, model: str):

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=0
        )

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0
        )

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=0
        )

    raise ValueError(f"Backend inconnu : {backend}")


def extract_ai_agent(sentence: str) -> Command:
    """Extracts the query for the AI agent from a sentence.

    Removes the AI agent trigger from the input sentence and returns
    a successful command containing the remaining text as the query.

    Args:
        sentence: The input sentence containing the AI agent trigger.

    Returns:
        A successful Command with the extracted sentence stored as
        the `query` argument.
    """

    sentence = sentence.replace(AI_AGENT_TRIGGERS[0][2:-2], '')

    return Command(
        argument={'query': sentence},
        status='ok'
    )
