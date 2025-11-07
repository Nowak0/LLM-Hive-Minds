import subprocess
from langchain.chat_models import init_chat_model
from Agent import Agent


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


model_name = "gpt-oss:20b"
check_ollama_model(model_name)
llm = init_chat_model(model=model_name, model_provider="ollama")

mind = Agent(llm)
mind.run_chatbot()

try:
    subprocess.run(["ollama", "stop", model_name]) # stop ollama – saves a lot of RAM
except Exception as e:
    pass