from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_react_agent, AgentExecutor
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import subprocess


def check_ollama_model(model_name: str):
    """Ensure an Ollama model is available locally; pull if missing"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if model_name in result.stdout:
            return
        else:
            subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while checking or pulling model '{model_name}':\n{e}")
        raise


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]


class Agent:
    def __init__(self, model_name: str, tools, role: str):
        self.model_name = model_name
        check_ollama_model(model_name)
        self.llm = ChatOllama(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.prompt = self._build_prompt(role)
        self.query = ""
        self.tools = tools
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)


    def _build_prompt(self, role: str):
        """It's supposed to be a private method. Creates a full prompt for the agent"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{role}\n{format_instructions}"),
                ("placeholder", "{chat_history}"),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}")
            ]
        ).partial(format_instructions=self.parser.get_format_instructions(), role=role)

        return prompt


    def set_query(self, query: str):
        """Sets the query that will be executed by the agent"""
        self.query = query


    def run_agent(self):
        """Runs the agent and prints a response"""
        if self.query == "":
            return None

        raw_response = self.agent_executor.invoke({"query": self.query})

        try:
            structured_response = self.parser.parse((raw_response.get("output") or raw_response.get("output_text")))
            subprocess.run(["ollama", "stop", self.model_name]) # stop ollama – saves A LOT of RAM
            return structured_response
        except Exception as e:
            subprocess.run(["ollama", "stop", self.model_name]) # stop ollama – saves A LOT of RAM
            return "Error parsing response:", e, "Raw Response: ", raw_response