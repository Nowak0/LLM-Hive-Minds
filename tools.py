from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime

def save_output_to_file(data: str, filename: str = "output.txt"):
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Output {timestamp} ---\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"Output saved to {filename}"

save_tool = Tool(
    name="save_output_to_file",
    func=save_output_to_file,
    description="Save output to file",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_web",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)