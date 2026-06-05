import json

from Pure.Agent import Agent
from Pure.exceptions import StatusMismatchException, StagnationException
from Pure.prompts import WORKER

CONSOLE_LOGS = False


def console_log(message):
    if CONSOLE_LOGS:
        print(message)


def run_agent(agent: Agent, insight: dict, temperature: float = None, max_tokens: int = None):
    """Build a proper prompt for given agent and runs a chat with it"""
    prompt = f"""
    USER_PROMPT: {insight.get('prompt')},

    RESEARCH: {insight.get('research')},

    CALCULATIONS: {insight.get('calculations')},

    FINAL_ANSWER: {insight.get('final_answer')}
    """
    prompt = agent.build_chat_prompt(prompt)

    kwargs = {}
    if temperature is not None:
        kwargs['temperature'] = temperature
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens

    return agent.ollama_chat(prompt=prompt, **kwargs)


def run_worker(agent: Agent, insight: dict, temperature: float, max_tokens: int):
    result = run_agent(agent=agent, insight=insight, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.loads(result)
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


def handle_new_information(information_arr: list[str], new_information: str):
    console_log(f"\tNew information: {new_information}")

    if new_information not in information_arr:
        information_arr.append(new_information)
    else:
        stagnation_counter = 0
        for i in information_arr[-10:]:
            if i == new_information:
                stagnation_counter += 1
        if stagnation_counter >= 4:
            raise StagnationException(f"Stagnation occurred: {new_information}")


def handle_response(response, insight: dict):
    console_log(f"\tThought process:\n\t- {response.get('thought')}")

    status = response.get('status')
    console_log(f"\tStatus: {status}")

    if status == 'research':
        if response.get('calculation') is not None:
            raise StatusMismatchException(f"Wrong status error. Response: {response.get('calculation')}")

        if response.get('research') is None:
            handle_new_information(insight.get('research'), response.get('thought'))
        else:
            handle_new_information(insight.get('research'), response.get('research'))
    elif status == 'calculating':
        if response.get('research') is not None:
            raise StatusMismatchException(f"Wrong status error. Response: {response.get('research')}")

        handle_new_information(insight.get('calculations'), response.get('calculation'))
    elif status == 'done':
        insight['final_answer'] = response.get('final_answer')
    elif status == 'idle':
        return
    else:
        raise ValueError(f"Unknown status response {status}")


def handle_worker(worker: Agent, insight: dict):
    """Runs worker with no role defined"""
    try:
        response = run_worker(worker, insight=insight, temperature=0.05, max_tokens=2000)
        return handle_response(response, insight)
    except StatusMismatchException as e:
        console_log(f"Worker responded with wrong status code. Correct response: {e}")
        return None


def prepare_workers(user_input: str, console_logs_bool: bool, n_workers: int, used_models: list):
    """Model with no roles defined (agents have the same roles)"""
    global CONSOLE_LOGS
    CONSOLE_LOGS = console_logs_bool

    workers = [Agent(model=used_models[i % len(used_models)], role=WORKER) for i in range(n_workers)]
    insight = {
        "prompt": user_input,
        "thoughts": [],
        "research": [],
        "calculations": [],
        "final_answer": None
    }

    console_log("START SOLVING")
    while insight.get('final_answer') is None:
        for idx, worker in enumerate(workers, start=1):
            console_log(f"Worker{idx}:")

            try:
                handle_worker(worker=worker, insight=insight)
                console_log(f"\tTotal research:\n\t\t{insight.get('research')}")
                console_log(f"\tTotal calculations:\n\t\t{insight.get('calculations')}\n")
            except StagnationException:
                print("Could not find final answer.")
                if insight.get('calculations'):
                    print("Last calculation: ", insight.get('calculations')[-1])
                return None

    return insight.get('final_answer')
