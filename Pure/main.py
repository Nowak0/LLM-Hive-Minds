import random
import json
import math
from Pure.Agent import Agent, check_ollama_model, quit_ollama

CALCULATION_RUNS = 9
MODEL = "llama3.1:8b"

ROLE_RESEARCHER = """You are a researcher that gathers insight about given math problem.

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
ROLE_CALCULATOR_BASE = """You are a math specialist that calculates equations. 

TASK:
Compute the final numeric/symbolic result for the given expression/problem.

OUTPUT (STRICT):
- Return EXACTLY one JSON object with exactly one key: "final_answer".
- Do NOT output any extra keys, text, markdown, or whitespace outside JSON.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- Use standard ASCII: "sqrt(2)" not "√2".
- If there are multiple valid solutions, output them as a JSON string with a deterministic separator: "x1; x2; ...".
- If there is no valid solution, output "#no_solution".

OUTPUT FORMAT:
{
    "final_answer": result
}
"""
ROLE_CALCULATOR_ALGEBRA = """You are a math calculation engine specialized in symbolic simplification.

METHOD:
1) Rewrite the problem into a single simplified symbolic expression.
2) Simplify algebraically (factor/cancel/common denominators) before evaluating.
3) Only at the end produce the final exact result.

OUTPUT (STRICT):
- Return EXACTLY one JSON object with exactly one key: "final_answer".
- Do NOT output any extra keys, text, markdown, or whitespace outside JSON.

FINAL_ANSWER RULES:
- Prefer exact forms: fractions, sqrt(), pi.
- Do not round unless explicitly requested.

FORMAT:
{
    "final_answer": result
}
"""
ROLE_CALCULATOR_STEPWISE = """You are a math calculation engine specialized in careful stepwise arithmetic.

METHOD:
1) Evaluate operations in a strict, explicit order (parentheses, powers, mult/div, add/sub).
2) Keep results as exact fractions whenever possible.
3) Perform cancellation and gcd reductions frequently to avoid overflow/mistakes.

OUTPUT (STRICT):
- Return EXACTLY one JSON object with exactly one key: "final_answer".
- Do NOT output any extra keys, text, markdown, or whitespace outside JSON.

FINAL_ANSWER RULES:
- Prefer exact fractions and symbolic constants.
- Do not round unless explicitly requested.

FORMAT:
{
    "final_answer": result
}
"""
ROLES_CALCULATOR = [ROLE_CALCULATOR_BASE, ROLE_CALCULATOR_ALGEBRA,ROLE_CALCULATOR_STEPWISE]
ROLE_EVALUATOR = """You are a STRICT result selector for a math task.

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

SELECTION POLICY:
1) Filter invalid/empty candidates (empty string, None-like, NaN-like). If everything is invalid -> "#not_good".
2) Group candidates into “equivalence clusters” using the normalization rules above.
3) Prefer the cluster with the largest support (most candidates that agree).
4) If there is a tie for largest support -> "#not_good".
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


def handle_worker(role: str, input: str, max_tokens: int):
    agent = Agent(model=MODEL, role=role)
    result = run_agent(agent=agent, input=input, temperature=random.uniform(0.03, 0.1), max_tokens=max_tokens)

    try:
        return json.loads(result).get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


def handle_calculations(evaluator: Agent, user_input: str, research: str, max_tokens: int):
    """Runs calculations with varying temperature"""
    results = []
    possible_answers = ""
    start_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    print("START CALCULATIONS")
    for _ in range(CALCULATION_RUNS):
        # role = random.choice(ROLES_CALCULATOR)
        idx = random.randint(1, len(ROLES_CALCULATOR))
        role = ROLES_CALCULATOR[idx - 1]
        result = handle_worker(role=role, input=start_input, max_tokens=max_tokens)
        possible_answers += f"\n- {result}"
        results.append(result)
        print(f"single calculation ({idx}): {result}")

    # results = normalize_results(results)

    while True:
        print("POSSIBLE ANSWERS: ", results)
        output_evaluation = handle_evaluation(agent=evaluator, user_input=user_input, research=research,
                                              results=results, temperature=0.05, max_tokens=100)
        print(output_evaluation)
        if output_evaluation != "#not_good":
            break
        else:
            print(f"Answers not good enough")
        full_input = f"""{start_input}

        POSSIBLE ANSWERS: {possible_answers}"""
        role = random.choice(ROLES_CALCULATOR)
        result = handle_worker(role=role, input=full_input, max_tokens=max_tokens)
        possible_answers += f"\n- {result}"
        results.append(result)
        print(f"single calculation ({role}): {result}")

    if len(results) < CALCULATION_RUNS // 2 + 1:
        raise Exception("Too many failed calculations")

    return output_evaluation


def handle_evaluation(agent: Agent, user_input, research: str,results: list, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = json.dumps({
        "question": user_input,
        "research": research,
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

    try:
        agent_researcher = Agent(model=MODEL, role=ROLE_RESEARCHER)
        agent_evaluator = Agent(model=MODEL, role=ROLE_EVALUATOR)
        # user_input = input("> ")
        user_input = "Find a rational approximation of π with denominator less than 1,000"

        research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.25, max_tokens=2000)
        print(research)

        results = handle_calculations(evaluator=agent_evaluator, user_input=user_input,
                                      research=research, max_tokens=150)

        print("AGENT EVALUATION: ", results)

    finally:
        quit_ollama(MODEL)


if __name__ == "__main__":
    main()
