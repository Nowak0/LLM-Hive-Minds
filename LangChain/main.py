from tools import *
from Agent import *

ollama_model = "gpt-oss:20b"
tools = [search_tool, wiki_tool, save_tool]
role = """
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text.
"""

research_agent = Agent(ollama_model, tools, role)
query = input("What would you like to search for? ")
research_agent.set_query(query)

response = research_agent.run_agent()
print(response)