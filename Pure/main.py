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

YOU WILL RECEIVE (as JSON in the user message):
- question: the original user question/expression
- research: factual, non-calculational insights relevant to the problem (may include constraints, domains, definitions, typical pitfalls)
- possible_results: an array of candidate final answers (strings or numbers)

GOAL:
Select the best final answer ONLY from possible_results.
If no candidate is reliable enough OR quality is insufficient, output "#not_good".

ABSOLUTE CONSTRAINTS:
- Output EXACTLY one JSON object with exactly one key: "final_answer".
- "final_answer" MUST be either:
  (a) one element copied from possible_results EXACTLY as it appears, or
  (b) the string "#not_good".
- Do NOT include any other keys, text, markdown, or explanations.
- Do NOT invent a new answer.
- Do NOT “improve” formatting, rounding, precision, or rewrite the chosen value.

CRITICAL PRINCIPLE:
Research is factual context. Use it to validate or invalidate candidates, NOT to compute a new answer.

SELECTION POLICY (STRICT ORDER):

0) Input hygiene:
   - Remove invalid candidates (empty string, null-like, NaN-like, non-scalar JSON objects/arrays).
   - If none remain -> "#not_good".

1) Extract explicit requirements from question/research (NO CALCULATION):
   If question/research explicitly requests any of the following, treat it as a requirement:
   - Precision requirement (examples):
     * "to 5 decimal places" / "5 digits after the decimal"
     * "rounded to N decimals"
     * significant figures
   - Format requirement:
     * must be an integer, fraction, simplified fraction, etc.
   - Domain requirement:
     * must be positive/real/in [a,b], etc.

2) Research/question hard constraint filtering (only when explicit and unambiguous):
   - Eliminate any candidate that clearly violates explicit constraints from question/research (domain, impossibility, format).
   - If all eliminated -> "#not_good".

3) Quality / satisfiability gate (soft-but-actionable):
   This gate is about whether a candidate is "good enough" for what the user asked, without recomputing.
   - If an explicit precision requirement exists:
     * Determine required decimals N (or significant figures) from question/research.
     * For each candidate, check if it satisfies the requirement by INSPECTION ONLY:
       - If candidate is a decimal string/number: count digits after '.' (or after ',' if used).
       - If candidate is in exact symbolic form (e.g., "pi", "22/7", "sqrt(2)"):
         - It does NOT satisfy a "N decimals" requirement unless the question explicitly allows exact forms instead of decimals.
     * If NO candidate satisfies the explicit precision requirement -> output "#not_good".
       (This is allowed and recommended to trigger recomputation with proper precision.)
   - If there is no explicit precision/format requirement, do not apply this gate.

4) Equivalence clustering:
   - Normalize candidates for comparison WITHOUT changing final output:
     * trim whitespace
     * treat numeric strings and numbers as comparable
     * cluster obvious near-equals (e.g., "42", "42.0", "41.999999999") within tiny tolerance
     * cluster obvious fraction/decimal equivalences when trivial (e.g., "0.5" vs "1/2")
   - Group into equivalence clusters.

5) Choose by support with research-aware tie-breaking:
   - Prefer the cluster with the largest support (most candidates that agree).
   - If tie:
     * If an explicit constraint (domain/format/precision) selects exactly one tied cluster, choose that one.
     * Otherwise -> "#not_good".

6) Final reliability gate:
   - If multiple distinct clusters remain and top cluster support < 2 -> "#not_good".
   - If results are scattered with no clear cluster -> "#not_good".
   - If only one valid candidate remains:
     * return it only if it meets explicit constraints and (if present) the precision requirement; else "#not_good".

OUTPUT FORMAT (exactly):
{
  "final_answer": "<one of possible_results or #not_good>"
}

EXAMPLE (precision):
question: "Approximate pi to 5 decimal places"
possible_results: ["3.14", "3.14159", "pi"]
-> choose "3.14159"
If possible_results: ["3.14", "pi"]
-> "#not_good"
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


def handle_calculations(evaluator: Agent, user_input: str, research: str, max_tokens: int, consol_logs: bool):
    """Runs calculations with varying temperature"""
    results = []
    possible_answers = ""
    start_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    print("START CALCULATIONS")
    for _ in range(CALCULATION_RUNS):
        if consol_logs:
            idx = random.randint(1, len(ROLES_CALCULATOR))
            role = ROLES_CALCULATOR[idx - 1]
        else:
            role = random.choice(ROLES_CALCULATOR)
        result = handle_worker(role=role, input=start_input, max_tokens=max_tokens)
        possible_answers += f"\n- {result}"
        results.append(result)
        if consol_logs:
            print(f"single calculation ({idx}): {result}")

    while True:
        if consol_logs:
            print("POSSIBLE ANSWERS: ", results)
        output_evaluation = handle_evaluation(agent=evaluator, user_input=user_input, research=research,
                                              results=results, temperature=0.05, max_tokens=100)
        if consol_logs:
            print(output_evaluation)

        if output_evaluation != "#not_good":
            break
        else:
            print(f"Answers not good enough")
        full_input = f"""{start_input}

        POSSIBLE ANSWERS: {possible_answers}"""
        if consol_logs:
            idx = random.randint(1, len(ROLES_CALCULATOR))
            role = ROLES_CALCULATOR[idx - 1]
        else:
            role = random.choice(ROLES_CALCULATOR)
        result = handle_worker(role=role, input=full_input, max_tokens=max_tokens)
        possible_answers += f"\n- {result}"
        results.append(result)
        if consol_logs:
            print(f"single calculation ({idx}): {result}")

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
        return output
    except KeyError:
        raise RuntimeError("Evaluation JSON missing 'final_answer' key")
    except ValueError:
        raise RuntimeError("Evaluator returned non-numeric answer")


def main():
    check_ollama_model(MODEL)
    consol_logs = False

    try:
        agent_researcher = Agent(model=MODEL, role=ROLE_RESEARCHER)
        agent_evaluator = Agent(model=MODEL, role=ROLE_EVALUATOR)
        user_input = input("> ")
        # user_input = "Find a rational approximation of π with denominator less than 1,000"

        research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.25, max_tokens=2000)
        if consol_logs:
            print(research)

        results = handle_calculations(evaluator=agent_evaluator, user_input=user_input,
                                      research=research, max_tokens=150, consol_logs=consol_logs)

        print("AGENT EVALUATION: ", results)

    finally:
        quit_ollama(MODEL)


if __name__ == "__main__":
    main()
