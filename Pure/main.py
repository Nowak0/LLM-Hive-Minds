import random
import json
import math
from Pure.Agent import Agent, check_ollama_model, quit_ollama


CALCULATION_RUNS = 9
MODEL = "llama3.1:8b"


def normalize_results(results):
    normalized_results = []

    for res in results:
        try:
            value = float(res["final_answer"])
            if math.isnan(value) or math.isinf(value):
                continue
            else:
                normalized_results.append(value)
        except Exception as e:
            print("Could not normalize calculated result:", e)
            continue

    if len(normalized_results) == 1:
        return normalized_results[0]
    else:
        return normalized_results


def run_agent(agent: Agent, input: str, temperature: float = None, max_tokens: int = None):
    """Build a proper prompt for given agent and runs a chat with it"""
    prompt = agent.build_chat_prompt(input)

    if temperature is not None and max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    elif temperature is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature)
    elif max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, max_tokens=max_tokens)
    else:
        return agent.ollama_chat(prompt=prompt)


def handle_research(agent: Agent, user_input, temperature: float, max_tokens: int):
    """Gathers insight from researcher and injects it into user's query"""
    raw_insight = run_agent(agent=agent, input=user_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.loads(raw_insight)
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Research agent failed to produce valid JSON")


def handle_calculations(role: str, user_input, research, max_tokens: int):
    """Runs calculations with varying temperature"""
    results = []
    full_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """

    agent = Agent(model=MODEL, role=role)
    for _ in range(CALCULATION_RUNS):
        result = run_agent(agent=agent, input=full_input, temperature=random.uniform(0.03, 0.1), max_tokens=max_tokens)
        results.append(json.loads(result))

    results = normalize_results(results)

    if len(results) < CALCULATION_RUNS // 2 + 1:
        raise Exception("Too many failed calculations")

    return results


def handle_evaluation(agent: Agent, user_input, results: list, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = json.dumps({
        "question": user_input,
        "possible_results": results
    }, indent=2)
    output = run_agent(agent=agent, input=new_input, temperature=temperature, max_tokens=max_tokens)

    try:
        output = json.loads(output)
        return normalize_results([output])
    except KeyError:
        raise RuntimeError("Evaluation JSON missing 'final_answer' key")
    except ValueError:
        raise RuntimeError("Evaluator returned non-numeric answer")


def main():
    check_ollama_model(MODEL)
    role_researcher = """You are a researcher that gathers insight about given math problem.
    TASK:
    Extract ONLY factual, non-calculational knowledge required to solve the problem. 
    Your main goal is to prepare a valuable information, basing it on:
    academic papers, studies, common knowledge, internet, etc.
    
    RULES:
    - Do not perform any calculations.
    - Do not solve the problem.
    - Be specific, be insightful, be thorough.
    - Do not provide any unnecessary text.
    - Do not speculate
    - Output MUST be a valid JSON
    """
    role_calculator = """You are a deterministic math specialist that calculates equations. 

    MAIN RULE:
    Answer in the following JSON format (very important!). Your final answer should be in "final_answer" key.
        
    RULES FOR FINAL_ANSWER:
    - The final_answer should return one result.
    - Output only the result of the given equation. 
    - Do not provide any additional text, just the result. No words, no units, no explanation. 
    - Use provided facts.
    - No formatting.
    - If there are fractions, return always decimal fractions, do not return common fractions.
    
    OUTPUT FORMAT:
    {
        "final_answer": result
    }
    """
    role_evaluator = """You are a math evaluator that receives many results of a certain math equation.
        TASK:
        Evaluate what the correct answer is out of the given array of possible answers.
        
        MAIN RULE:
        Answer in the following format (very important!). Your final answer should be in "final_answer" key.
        
        RULES:
        - Return only correct results. 
        - Base your answer on those provided in the input!!.
        - Do not provide a result that is not in the provided input!!. 
        - If one result is consistent in the input, choose it. 
        - If there are many results similar to each other, try to choose one of those, instead of the one that is utterly different.
        - Provide only the answer without any explanation or additional text. 
        - Do not to calculate, evaluate!
        
        OUTPUT FORMAT:
        {
            "final_answer": result
        }
        """
    
    try:
        agent_researcher = Agent(model=MODEL, role=role_researcher)
        agent_evaluator = Agent(model=MODEL, role=role_evaluator)
        user_input = input("> ")

        research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.25, max_tokens=2000)
        results = handle_calculations(role=role_calculator, user_input=user_input, research=research, max_tokens=150)
        print("POSSIBLE ANSWERS: ", results)
        output_evaluation_1 = handle_evaluation(agent=agent_evaluator, user_input=user_input,
                                 results=results, temperature=0.05, max_tokens=100)
        print("AGENT EVALUATION: ", output_evaluation_1)

    finally:
        quit_ollama(MODEL)

if __name__ == "__main__":
    main()
