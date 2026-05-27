import subprocess
import requests
import aiohttp

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 120


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
        print(e)


class Agent():
    def __init__(self, model, role):
        self.model = model
        self.role = role

    def build_chat_prompt(self, user_input):
        """Build a chat prompt"""
        return [
            {"role": "system", "content": self.role},
            {"role": "user", "content": user_input}
        ]

    async def ollama_chat(self, prompt: list[dict], temperature: float = 0.7, max_tokens: int = 2000):
        """Get a response from Ollama /api/chat"""
        package = {
            "model": self.model,
            "messages": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "no_cache": True,
            },
            "stream": False,
            "format": "json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_CHAT_URL, json=package) as response:
                response.raise_for_status()
                data = await response.json()
                return data["message"]["content"]
