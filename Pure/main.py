import random
import json

from Pure.Agent import Agent, check_ollama_model, quit_ollama
from Pure.exceptions import StagnationException, StatusMismatchException
from Pure.prompts import WORKER
from Pure.same_role_workers import prepare_workers, console_log

MODEL_OLD = "llama3.1:8b"
MODEL_HEAVY = "deepseek-r1:14b"
MODEL_REGULAR_1 = "qwen2.5:7b"
MODEL_REGULAR_2 = "gemma2:9b"
MODEL_LIGHT_ANALYTICAL = "phi4-mini"
MODEL_LIGHT_KNOWLEDGE = "gemma2:2b"
CONSOLE_LOGS = True
N_WORKERS = 2
USED_MODELS = [MODEL_REGULAR_1, MODEL_REGULAR_2]


def main():
    try:
        for model in USED_MODELS:
            check_ollama_model(model)

        user_input = input("> ")
        final_answer = prepare_workers(user_input, CONSOLE_LOGS, N_WORKERS, USED_MODELS)
        print("\n\nFINAL ANSWER:", final_answer, "\n\n")
    finally:
        for model in USED_MODELS:
            quit_ollama(model)
        if CONSOLE_LOGS:
            print(f"Finished shutting down all used models")

if __name__ == "__main__":
    main()
