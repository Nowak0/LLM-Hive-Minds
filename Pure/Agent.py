import subprocess
import requests


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def check_ollama_model(model: str):
    """Ensure an Ollama model is available locally; pull if missing"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if model in result.stdout:
            return
        else:
            subprocess.run(["ollama", "pull", model], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while checking or pulling model '{model}':\n{e}")
        raise


def quit_ollama(model: str):
    """Quit Ollama. Saves a lot of RAM"""
    try:
        subprocess.run(["ollama", "stop", model])
    except Exception as e:
        pass


class Agent():
    def __init__(self, model, role):
        self.model = model
        self.role = role


    def build_chat_prompt(self, user_input: str):
        """Build a chat prompt"""
        return [
            {"role": "system", "content": self.role},
            {"role": "user", "content": user_input}
        ]


    def ollama_generate(self, role: str, user_input: str, temperature: float = 0.7, max_tokens: int = 300):
        """Get a response from Ollama /api/generate"""
        prompt = f"SYSTEM:{role}   USER:{user_input}   ASSISTANT:"
        package = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }

        response = requests.post(OLLAMA_GENERATE_URL, json=package)
        response.raise_for_status()
        return response.json()["response"]


    def ollama_chat(self, prompt: list[dict], temperature: float = 0.7, max_tokens: int = 300):
        """Get a response from Ollama /api/chat"""
        package = {
            "model": self.model,
            "messages": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }

        response = requests.post(OLLAMA_CHAT_URL, json=package)
        response.raise_for_status()
        return response.json()["message"]["content"]