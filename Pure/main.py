import random
import json

from Pure.Agent import Agent, check_ollama_model, quit_ollama
from Pure.exceptions import StagnationException, StatusMismatchException
from Pure.prompts import ROLES_CALCULATOR, ROLE_EVALUATOR, ROLE_RESEARCHER, NO_ROLE_WORKER

CALCULATION_RUNS = 3
MODEL_OLD = "llama3.1:8b"
MODEL_HEAVY = "deepseek-r1:14b"
MODEL_REGULAR = "qwen2.5:7b"
MODEL_LIGHT_ANALYTICAL = "phi4-mini"
MODEL_LIGHT_KNOWLEDGE = "gemma2:2b"
CONSOLE_LOGS = True

N_NO_ROLE_WORKERS = 2

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
        return json.dumps(json.loads(raw_insight), indent=2)
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Research agent failed to produce valid JSON")
    finally:
        quit_ollama(agent.model)


def run_worker(role: str, input: str, model: str, max_tokens: int):
    agent = Agent(model=model, role=role)
    result = run_agent(agent=agent, input=input, temperature=0.05, max_tokens=max_tokens)

    try:
        data = json.loads(result)
        if CONSOLE_LOGS:
            print(f"Worker thought:\n{data.get('thought')}\n")
        return data.get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


def handle_worker(start_input: str, possible_results: str, model: str,max_tokens: int, number_of_runs: int = 1):
    for _ in range(number_of_runs):
        idx = random.randint(0, len(ROLES_CALCULATOR)-1)
        role = ROLES_CALCULATOR[idx]
        result = run_worker(role=role, input=start_input, model=model, max_tokens=max_tokens)
        possible_results += f"\n- {result}"

        if CONSOLE_LOGS:
            print(f"single calculation ({idx+1}): {result}")

    quit_ollama(model)
    return possible_results


def handle_calculations(evaluator: Agent, user_input: str, research: str, max_tokens: int):
    """Runs calculations with varying temperature"""
    possible_results = ""
    output_evaluation = ""
    start_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    if CONSOLE_LOGS:
        print("START CALCULATIONS")

    possible_results = handle_worker(start_input=start_input, possible_results=possible_results, model=MODEL_HEAVY,
                                     max_tokens=max_tokens, number_of_runs=CALCULATION_RUNS)
    count_runs = 0
    while count_runs < CALCULATION_RUNS*3:
        if CONSOLE_LOGS:
            print("POSSIBLE ANSWERS: ", possible_results)

        output_evaluation = handle_evaluation(agent=evaluator, user_input=user_input, research=research,
                                              results=possible_results, temperature=0.05, max_tokens=1000)
        if CONSOLE_LOGS:
            print("evaluation: ", output_evaluation)

        if output_evaluation != "#not_good":
            break
        elif CONSOLE_LOGS :
            print(f"Answers not good enough")

        full_input = f"""{start_input}

        POSSIBLE ANSWERS: {possible_results}"""
        possible_results = handle_worker(start_input=full_input, possible_results=possible_results, model=MODEL_HEAVY,
                                         max_tokens=max_tokens, number_of_runs=1)
        count_runs += 1

    if count_runs >= CALCULATION_RUNS*3:
        raise Exception("Could not find reliable answer")

    quit_ollama(evaluator.model)
    return output_evaluation


def handle_evaluation(agent: Agent, user_input, research: str, results: str, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    
    POSSIBLE ANSWERS: {results}
    """

    output = run_agent(agent=agent, input=new_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.loads(output).get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")
    except KeyError:
        raise RuntimeError("Evaluation JSON missing 'final_answer' key")


def different_roles_defined(user_input: str):
    """Model where different roles for agents are defined (researcher, calculator, evaluator)"""
    agent_researcher = Agent(model=MODEL_LIGHT_ANALYTICAL, role=ROLE_RESEARCHER)
    agent_evaluator = Agent(model=MODEL_REGULAR, role=ROLE_EVALUATOR)
    user_input = input("> ")

    research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.15, max_tokens=2000)
    if CONSOLE_LOGS:
        print(research)

    results = handle_calculations(evaluator=agent_evaluator, user_input=user_input,
                                  research=research, max_tokens=2000)

    print("AGENT EVALUATION: ", results)


def run_basic_worker(agent: Agent, input: str, temperature: float, max_tokens: int):
    result = run_agent(agent=agent, input=input, temperature=temperature, max_tokens=max_tokens)

    try:
        data = json.loads(result)
        # if CONSOLE_LOGS:
        #     print(f"Worker thought:\n{data.get('thought')}\n")
        return data
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


def handle_new_information(information_arr: list[str], new_information: str):
    if CONSOLE_LOGS:
        print(f"\t{new_information}")

    if new_information not in information_arr:
        information_arr.append(new_information)
    else:
        stagnation_counter = 0
        for i in information_arr[-10:]:
            if i == new_information:
                stagnation_counter += 1
        if stagnation_counter >= 6:
            raise StagnationException(f"Stagnation occurred: {new_information}")



def handle_response(response, research: list[str], calculations: list[str], final_answer):
    if CONSOLE_LOGS:
        print(f"\tThought process:\n\t- {response.get("thought")}")

    found_final_answer = False

    status = response.get('status')
    if CONSOLE_LOGS:
        print(f"\tStatus: {status}")
    match status:
        case "research":
            handle_new_information(research, response.get("research"))
            if response.get("calculation") is not None:
                raise StatusMismatchException(f"Wrong status error. Response: {response.get("calculation")}")
        case "calculating":
            handle_new_information(calculations, response.get("calculation"))
            if response.get("research") is not None:
                raise StatusMismatchException(f"Wrong status error. Response: {response.get("research")}")
        case "done":
            final_answer = response.get("final_answer")
            found_final_answer = True
        case "idle":
            found_final_answer = True
        case _:
            pass

    return found_final_answer, final_answer


def handle_basic_worker(worker: Agent, user_input: str, research: list[str], calculations: list[str], final_answer):
    """Runs worker with no role defined"""
    formatted_research = "\n".join([f"- {item}" for item in research]) if research else "None"
    formatted_calculations = "\n".join([f"- {item}" for item in calculations]) if calculations else "None"
    input = f"""
    QUESTION: {user_input}

    RESEARCH: \n{formatted_research}
    
    CALCULATIONS: \n{formatted_calculations}
    """
    try:
        response = run_basic_worker(worker, input=input, temperature=0.05, max_tokens=2000)
        return handle_response(response, research, calculations, final_answer)
    except StatusMismatchException:
        print("Worker responded with wrong status code. Correct response:")
        print(StatusMismatchException)


def no_roles_defined(user_input: str):
    """Model with no roles defined (agents have the same roles)"""
    workers = [Agent(model=MODEL_REGULAR, role=NO_ROLE_WORKER) for _ in range(N_NO_ROLE_WORKERS)]
    found_final_answer = False
    final_answer = None
    research = []
    calculations = []
    if CONSOLE_LOGS:
        print("START SOLVING")
    while not found_final_answer:
        idx = 1
        for worker in workers:
            if CONSOLE_LOGS:
                print(f"Worker{idx}:")
            try:
                found_final_answer, final_answer = handle_basic_worker(worker=worker, user_input=user_input,
                                                                   research=research, calculations=calculations,
                                                                   final_answer=final_answer)
            except StagnationException:
                print("Could not find final answer.")
                if calculations:
                    print("Last calculation: ", calculations[-1])
            idx += 1

    print("FINAL ANSWER: ", final_answer)


def main():
    check_ollama_model(MODEL_LIGHT_ANALYTICAL)
    check_ollama_model(MODEL_REGULAR)
    check_ollama_model(MODEL_HEAVY)

    try:
        user_input = input("> ")
        # if CONSOLE_LOGS:
        #     print("User input: ", user_input)

        # different_roles_defined(user_input)
        no_roles_defined(user_input)
    finally:
        quit_ollama(MODEL_LIGHT_ANALYTICAL)
        quit_ollama(MODEL_REGULAR)
        quit_ollama(MODEL_HEAVY)


if __name__ == "__main__":
    main()
