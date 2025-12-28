import os
from concurrent.futures.thread import ThreadPoolExecutor
import random
import json
from Pure.Agent import Agent, check_ollama_model, quit_ollama


MAX_WORKERS = min(3, os.cpu_count())
CALCULATION_RUNS = 10
MODEL = "llama3.2"


def run_agent(agent: Agent, user_input: str, temperature: float = None, max_tokens: int = None):
    """Build a proper prompt for given agent and runs a chat with it"""
    prompt = agent.build_chat_prompt(user_input)

    if temperature is not None and max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    elif temperature is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature)
    elif max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, max_tokens=max_tokens)
    else:
        return agent.ollama_chat(prompt=prompt)


def handle_research(agent_researcher: Agent, user_input):
    """Gathers insight from researcher and injects it into user's query"""
    insight = run_agent(agent=agent_researcher, user_input=user_input, temperature=0.8, max_tokens=450)
    user_input += "\n\n RESEARCH:\n" + insight

    return user_input


def handle_calculations(role_calculator: str, user_input):
    """Runs calculations in parallel, creates a new agent object for every thread"""
    answers = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run_agent,
                            Agent(model=MODEL, role=role_calculator),
                            user_input,
                            random.uniform(0.2, 0.8))
            for _ in range(CALCULATION_RUNS)
        ]

        try:
            for f in futures:
                answers.append(f.result(timeout=60))
        except Exception as e:
            print("Agent failed: ", e)

    print("POSSIBLE ANSWERS: ", answers)
    return answers


def handle_evaluation(agent_evaluator: Agent, user_input, answers: list, temperature: float = None, max_tokens: int = None):
    """Adds possible results to user's query and evaluates them"""
    new_input = json.dumps({
        "question": user_input,
        "possible_results": answers
    }, indent=2)
    output = run_agent(agent=agent_evaluator, user_input=new_input, temperature=0.4, max_tokens=150)

    return str(output)


def main():
    check_ollama_model(MODEL)
    role_researcher = """You are a researcher that gathers insight about given math problem.
    Your main goal is to prepare a valuable information, basing it on:
    academic papers, studies, common knowledge, internet, etc.
    Under no circumstances can you carry out calculations, 
    your job is only to provide necessary knowledge.
    Be specific, be insightful, be thorough.
    Do not provide any unnecessary text.
    Absolutely do not provide a result or an answer.
    """
    role_calculator = """You are a math specialist that calculates different equations. 
    Give only the result of the given equation. No text, just result. If given any insight,
    use it."""
    role_evaluator = """You are an assessor that receives many results of a certain math equation
    and evaluates what the correct answer is. Return only correct results. Base your answer on those provided in input.
    Do not provide a result that is not in the provided input UNLESS all answers in input are incorrect. 
    If one result is consistent in the input (and correct), choose it. 
    Provide only the answer without any explanation. Your job is not to calculate, your job is to evaluate"""
    agent_researcher = Agent(model=MODEL, role=role_researcher)
    agent_evaluator = Agent(model=MODEL, role=role_evaluator)

    user_input = input("> ")

    user_input = handle_research(agent_researcher, user_input)
    answers = handle_calculations(role_calculator, user_input)

    output_evaluation_1 = handle_evaluation(agent_evaluator, user_input, answers)
    print("ANSWER AFTER FIRST EVALUATION: ", output_evaluation_1)

    output_evaluation_2 = handle_evaluation(agent_evaluator, user_input, answers)
    print("ANSWER AFTER SECOND EVALUATION: ", output_evaluation_2)

    quit_ollama(MODEL)


if __name__ == "__main__":
    main()
