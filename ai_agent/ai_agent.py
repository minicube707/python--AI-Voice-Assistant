from nlp.command import Command
from nlp.intent import AI_AGENT_TRIGGERS

from langchain.agents import create_agent
from ai_agent.tools import search_tool, wiki_tool, get_weather


class AIAgent:

    def __init__(self, backend: str, model: str):
        
        tools = [
            search_tool,
            wiki_tool,
            get_weather
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
