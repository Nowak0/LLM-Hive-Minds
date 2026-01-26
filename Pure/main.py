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
            # normalized_results.append(value)
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


def handle_calculations(role: str, evaluator: Agent, user_input, research, max_tokens: int):
    """Runs calculations with varying temperature"""
    results = []
    full_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    print("START CALCULATIONS")
    agent = Agent(model=MODEL, role=role)
    for _ in range(CALCULATION_RUNS):
        result = run_agent(agent=agent, input=full_input, temperature=random.uniform(0.03, 0.1), max_tokens=max_tokens)
        results.append(json.loads(result))
        print(f"single calculation: {json.loads(result)}")

    results = normalize_results(results)

    while (True):
        print("POSSIBLE ANSWERS: ", results)
        output_evaluation = handle_evaluation(agent=evaluator, user_input=user_input,
                                              results=results, temperature=0.05, max_tokens=100)
        print(output_evaluation)
        if output_evaluation != "#not_good":
            break
        else:
            print(f"Answers not good enough")
        result = run_agent(agent=agent, input=full_input, temperature=random.uniform(0.03, 0.1), max_tokens=max_tokens)
        results.append(normalize_results([json.loads(result)]))
        print(f"single calculation: {json.loads(result)}")

    if len(results) < CALCULATION_RUNS // 2 + 1:
        raise Exception("Too many failed calculations")

    return output_evaluation


def handle_evaluation(agent: Agent, user_input, results: list, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = json.dumps({
        "question": user_input,
        "possible_results": results
    }, indent=2)
    output = run_agent(agent=agent, input=new_input, temperature=temperature, max_tokens=max_tokens)

    try:
        output = json.loads(output).get("final_answer")
        # return normalize_results([output])
        return output
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
    role_evaluator = """You are a STRICT result selector for a math task.
        
        You will receive:
        - a list/array called possible_results containing candidate final answers (strings or numbers).
        - optionally: the original question/expression (may be present, may be absent).
        
        GOAL:
        Select the best final answer ONLY from possible_results.
        If no candidate is reliable enough, output "#not_good".
        
        ABSOLUTE CONSTRAINTS:
        - You MUST output exactly one JSON object with one key: "final_answer".
        - "final_answer" MUST be either:
          (a) one element copied from possible_results (exactly as it appears), or
          (b) the string "#not_good".
        - Do NOT include any other keys, text, markdown, or explanations.
        - Do NOT invent a new answer.
        - Do NOT “improve” formatting or rounding of the chosen value.
        
        INTERPRETATION RULES (how to judge sameness without doing full math):
        You are allowed to do ONLY light consistency checks and normalization to compare candidates:
        - Trim whitespace.
        - Treat commas as decimal separators only if clearly numeric (e.g., "3,14" -> 3.14 for comparison).
        - Ignore surrounding quotes for comparison.
        - Consider numeric forms equivalent for grouping:
          - integers vs floats: "2" ~ "2.0"
          - scientific notation: "1e3" ~ "1000"
          - fractions vs decimals when obviously equal: "1/2" ~ "0.5"
          - "-0" ~ "0"
        - For numeric comparison, use a tolerance:
          - relative tol = 1e-6 OR absolute tol = 1e-9 (whichever is larger).
        - If answers include units/text (e.g., "42 m", "x=42"), treat the numeric part as the comparable value ONLY if the rest is consistent across multiple candidates. Otherwise treat them as different.
        
        SELECTION POLICY:
        1) Filter invalid/empty candidates (empty string, None-like, NaN-like). If everything is invalid -> "#not_good".
        2) Group candidates into “equivalence clusters” using the normalization rules above.
        3) Prefer the cluster with the largest support (most candidates that agree).
        4) If there is a tie for largest support:
           - Prefer the cluster that is most “plain final form” (a single number or simplest expression),
             over verbose forms like "x = 2", "Answer: 2", etc.
           - If still tied, choose the earliest appearing candidate among tied clusters.
        5) Reliability gate:
           - If the largest cluster has support < 2 AND there is noticeable disagreement (multiple distinct clusters), output "#not_good".
           - If the candidates are all over the place (no clear cluster), output "#not_good".
           - If possible_results contains only one candidate, return it ONLY if it looks like a valid final answer; otherwise "#not_good".
        
        OUTPUT FORMAT (exactly):
        {
          "final_answer": "<one of possible_results or #not_good>"
        }
        
        EXAMPLES:
        If possible_results = ["42", "42.0", "41.999999999"] -> choose "42" (or the earliest in that top cluster).
        If possible_results = ["10", "12", "9.5", "11"] -> "#not_good".
        """

    try:
        agent_researcher = Agent(model=MODEL, role=role_researcher)
        agent_evaluator = Agent(model=MODEL, role=role_evaluator)
        # user_input = input("> ")
        user_input = "Find a rational approximation of π with denominator less than 1,000"

        research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.25, max_tokens=2000)
        print(research)
        #         research = """
        # {"Archimedes' Method": {'Key Insight': 'Archimedes used the Pythagorean theorem to estimate the area and perimeter of polygons inscribed within a circle.', 'Rational Approximations': [{'Approximation': '3', 'Denominator': 1}, {'Approximation': '22/7', 'Denominator': 7}]}, 'Madhava-Leibniz Series': {'Key Insight': 'This series is a infinite expansion of π in terms of the arctangent function.', 'Rational Approximations': [{'Approximation': '3.141592653589793', 'Denominator': 1000}]}, 'Babylonian Method': {'Key Insight': 'This method uses an iterative formula to improve an initial estimate of π.', 'Rational Approximations': [{'Approximation': '3.14159', 'Denominator': 100}]}}
        # """
        results = handle_calculations(role=role_calculator, evaluator=agent_evaluator, user_input=user_input,
                                      research=research, max_tokens=150)
        # print("POSSIBLE ANSWERS: ", results)
        # output_evaluation_1 = handle_evaluation(agent=agent_evaluator, user_input=user_input,
        #                                         results=results, temperature=0.05, max_tokens=100)
        print("AGENT EVALUATION: ", results)

    finally:
        quit_ollama(MODEL)


if __name__ == "__main__":
    main()
