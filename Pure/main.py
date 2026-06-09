import random
import json
import asyncio
import time
from datetime import datetime

from Pure.Agent import Agent, check_ollama_model, quit_ollama
from questions.question_bank import get_chosen_question
EVALUATION_RUNS=2
CALCULATION_RUNS = 3
MODEL_OLD = "llama3.1:8b"
# MODEL_HEAVY = "deepseek-r1:14b"
MODEL_HEAVY = "llama3.1:8b"
MODEL_REGULAR = "qwen2.5:7b"
MODEL_REGULAR_LIGHT = "qwen2.5:3b"
MODEL_LIGHT_ANALYTICAL = "phi4-mini"
MODEL_LIGHT_KNOWLEDGE = "gemma2:2b"

# these models aren't the best at math but they are small an make testing easier/possible at all
CALCULATOR_MODELS = [MODEL_LIGHT_ANALYTICAL, MODEL_LIGHT_KNOWLEDGE, MODEL_REGULAR_LIGHT]

CONSOLE_LOGS = True
QUESTION_BANK = False #True

ROLE_RESEARCHER = """You are a researcher that gathers insight about given math problem.

TASK:
Provide (A) theory/method notes and (B) optional checkable constraints that help downstream agents.
You must NOT compute the specific solution. 
Your output MUST help downstream agents (calculators + evaluator) by providing:
1) factual background / methods (theory), AND
2) explicit, checkable constraints ONLY WHEN they are warranted (implied by the question and by standard definitions).

ABSOLUTE RULES:
- Do NOT perform any calculations.
- Do NOT solve the problem.
- Do NOT approximate any numeric answer for the specific instance.
- Do NOT select among candidate answers.
- Do NOT speculate or add heuristics.
- Output MUST be valid JSON only (no markdown, no extra text).

CONSTRAINT POLICY (VERY IMPORTANT):
- It is OK to output mostly null/empty constraints.
- You may fill a constraint ONLY if:
  (1) it is explicitly stated in the question, OR
  (2) it is a direct consequence of a standard definition that you explicitly wrote in theory_notes/methods.
- If you set ANY constraint field to a non-null value, add a corresponding entry to constraint_sources.

QUALITY REQUIREMENT:
- theory_notes MUST contain at least 3 items.
- methods MUST contain at least 1 item when the problem is non-trivial.
- constraints.validity_checks_for_evaluator should be present even if empty ([]).

REMINDER:
No numeric evaluation of candidates. No choosing “best fraction”. No computing convergents.
"""
ROLE_CALCULATOR_BASE = """You are a math specialist that calculates equations. 

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field without any reasoning.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

OUTPUT FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLE_CALCULATOR_ALGEBRA = """You are a math calculation engine specialized in symbolic simplification.

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

METHOD:
1) Rewrite the problem into a single simplified symbolic expression.
2) Simplify algebraically (factor/cancel/common denominators) before evaluating.
3) Only at the end produce the final exact result.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLE_CALCULATOR_STEPWISE = """You are a math calculation engine specialized in careful stepwise arithmetic.

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

METHOD:
1) Evaluate operations in a strict, explicit order (parentheses, powers, mult/div, add/sub).
2) Keep results as exact fractions whenever possible.
3) Perform cancellation and gcd reductions frequently to avoid overflow/mistakes.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLES_CALCULATOR = [ROLE_CALCULATOR_BASE, ROLE_CALCULATOR_ALGEBRA, ROLE_CALCULATOR_STEPWISE]
ROLE_EVALUATOR = """You are a STRICT result selector for a math task.

YOU WILL RECEIVE (as JSON in the user message):
- question: the original user question/expression
- research: factual, non-calculational insights relevant to the problem 
    (may include constraints, domains, definitions, typical pitfalls)
- possible_results: an array of candidate final answers (strings or numbers)

GOAL:
Select the best final answer ONLY from possible_results.
If no candidate is good enough OR quality is insufficient, output "#not_good".

ABSOLUTE CONSTRAINTS:
- Output EXACTLY one JSON object with exactly one key: "final_answer".
- "final_answer" MUST be either:
  (a) one element copied from possible_results EXACTLY as it appears, or
  (b) the string "#not_good".
- Do not explain and do not fix formatting
- NEVER accept a solution that is in a non-final form, 
    example: (-2 +- sqrt(20/3) * 18) or (10 choose 5) * 0.5^5 * 0.5^(10-5), etc.
    In such solutions it is better to return output "#not_good".

SELECTION POLICY (STRICT ORDER):

1) Filter: Remove invalid/null entries. Discard candidates violating 'research' (domain, precision, format).
2) Cluster: Group equivalent values (e.g., "0.5", "1/2", "0.50").
3) Evaluate: 
   - If a specific precision (e.g., "5 decimals") is required and no candidate meets it -> "#not_good".
   - If no precision specified prefer fractions
   - Pick the cluster with the most support.
   - If there is a tie or no clear majority (support < 2) and the problem is non-trivial -> "#not_good".

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


async def run_agent(agent: Agent, input: str, temperature: float = None, max_tokens: int = None):
    """Build a proper prompt for given agent and runs a chat with it"""
    prompt = agent.build_chat_prompt(input)

    if temperature is not None and max_tokens is not None:
        return await agent.ollama_chat(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    elif temperature is not None:
        return await agent.ollama_chat(prompt=prompt, temperature=temperature)
    elif max_tokens is not None:
        return await agent.ollama_chat(prompt=prompt, max_tokens=max_tokens)
    else:
        return await agent.ollama_chat(prompt=prompt)


async def handle_research(agent: Agent, user_input, temperature: float, max_tokens: int):
    """Gathers insight from researcher and injects it into user's query"""
    raw_insight = await run_agent(agent=agent, input=user_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.dumps(json.loads(raw_insight), indent=2)
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Research agent failed to produce valid JSON")
    finally:
        quit_ollama(agent.model)


async def run_worker(role: str, input: str, model: str, max_tokens: int):
    if CONSOLE_LOGS:
        start = datetime.now()
        print(f"[START] {role[9:27]}... at {start.strftime('%H:%M:%S')} for model: {model}")

    agent = Agent(model=model, role=role)
    result = await run_agent(agent=agent, input=input, temperature=0.05, max_tokens=max_tokens)

    if CONSOLE_LOGS:
        end = datetime.now()
        print(f"[END] {role[9:27]}... at {start.strftime('%H:%M:%S')} (duration {(end - start).total_seconds():.2f}s)")

    try:
        data = json.loads(result)
        if CONSOLE_LOGS:
            print(f"Worker thought:\n{data.get('thought')}\n")
        return data.get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


async def handle_worker(start_input: str, possible_results: str, max_tokens: int, number_of_runs: int = 1):
    tasks = []
    for _ in range(number_of_runs):
        idx = random.randint(0, len(ROLES_CALCULATOR) - 1)
        role = ROLES_CALCULATOR[idx]
        # if the models are the same for two calculators then we have a bottleneck and they're done sequentially anyway
        chosen_model = random.choice(CALCULATOR_MODELS)
        tasks.append(run_worker(role=role, input=start_input, model=chosen_model, max_tokens=max_tokens))

    results = await asyncio.gather(*tasks)
    #results = []  # it should be gather but this lessens the chances of a timeout for now and makes it actually possible to test
    #for t in tasks:
    #    results.append(await t)

    if CONSOLE_LOGS:
        for idx, result in enumerate(results):
            print(f"single calculation ({idx + 1}): {result}")

    #quit_ollama(model)
    return results


async def handle_calculations(evaluator: Agent, user_input: str, research: str, max_tokens: int):
    """Runs calculations with varying temperature"""
    possible_results = ""
    output_evaluation = ""
    start_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    if CONSOLE_LOGS:
        print("START CALCULATIONS")

    results_list = await handle_worker(start_input=start_input, possible_results=possible_results,
                                       max_tokens=max_tokens, number_of_runs=CALCULATION_RUNS)
    # possible_results = "\n".join(f"- {r}" for r in results_list)
    possible_results = results_list

    count_runs = 0
    while count_runs < CALCULATION_RUNS * 3:
        if CONSOLE_LOGS:
            print("POSSIBLE ANSWERS: \n", "\n".join(f"- {r}" for r in possible_results))
        tasks =[]
        for _ in range(EVALUATION_RUNS):
            tasks.append(handle_evaluation(agent=evaluator, user_input=user_input, research=research,
                                           results=possible_results, temperature=0.05, max_tokens=1000))
        output_evaluation = await asyncio.gather(*tasks)
        output_evaluation = await handle_answer(output_evaluation)

        if CONSOLE_LOGS:
            print("evaluation: ", output_evaluation)

        if output_evaluation != "#not_good":
            break
        elif CONSOLE_LOGS:
            print(f"Answers not good enough")

        full_input = f"""{start_input}

        POSSIBLE ANSWERS: {results_list}"""
        new_results = await handle_worker(start_input=full_input, possible_results=possible_results,
                                          max_tokens=max_tokens, number_of_runs=1)

        possible_results += "\n".join(f"- {r}" for r in new_results)
        count_runs += 1

    if count_runs >= CALCULATION_RUNS * 3:
        raise Exception("Could not find reliable answer")

    quit_ollama(evaluator.model)
    return output_evaluation

async def handle_answer(output_evaluation:str):
    final_answer = output_evaluation[0]
    i=1
    for answer in output_evaluation:
        if CONSOLE_LOGS:
            print(f"Answer from evaluator {i}: {answer}")
        if final_answer != answer:
            return "#not_good"
    return final_answer
async def handle_evaluation(agent: Agent, user_input, research: str, results: str, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}

    POSSIBLE ANSWERS: {results}
    """

    output = await run_agent(agent=agent, input=new_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.loads(output).get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")
    except KeyError:
        raise RuntimeError("Evaluation JSON missing 'final_answer' key")


async def main():
    check_ollama_model(MODEL_LIGHT_ANALYTICAL)
    check_ollama_model(MODEL_REGULAR)
    for model in CALCULATOR_MODELS:
        check_ollama_model(model)

    try:
        agent_researcher = Agent(model=MODEL_LIGHT_ANALYTICAL, role=ROLE_RESEARCHER)
        agent_evaluator = Agent(model=MODEL_REGULAR, role=ROLE_EVALUATOR)
        if QUESTION_BANK:
            question_input = get_chosen_question()
            print(f"Chosen question: {question_input}\n")
        else:
            question_input = input("> ")

        research = await handle_research(agent=agent_researcher, user_input=question_input, temperature=0.15,
                                         max_tokens=2000)
        if CONSOLE_LOGS:
            print(research)

        results = await handle_calculations(evaluator=agent_evaluator, user_input=question_input,
                                            research=research, max_tokens=3000)

        print("AGENT EVALUATION: ", results)

    finally:
        quit_ollama(MODEL_LIGHT_ANALYTICAL)
        quit_ollama(MODEL_REGULAR)
        quit_ollama(MODEL_HEAVY)


if __name__ == "__main__":
    asyncio.run(main())