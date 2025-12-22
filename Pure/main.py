from Pure.Agent import Agent, check_ollama_model, quit_ollama
import time

model = "llama3.2"
check_ollama_model(model)

role_calculator = """You are a math specialist that calculates different equations. 
Give only the result of the given equation. No text, just result"""

role_assessor = """You are an assessor that receives many results of a certain math equation
and evaluates what the correct answer is. Return only correct results.
Provide only the answer without any explanation. Your job is not to calculate, your job is to evaluate"""

user_input = input("> ")
answers = []
agent_c = Agent(model=model, role=role_calculator)
prompt = agent_c.build_chat_prompt(user_input)

for _ in range(10):
    output = agent_c.ollama_chat(prompt, temperature=0.4)
    answers.append(output)
    time.sleep(0.1)
print("POSSIBLE ANSWERS: ", answers)
print("\n\n")

agent_a = Agent(model=model, role=role_assessor)
new_input = "question: " + user_input + "\n\n array of possible results: " + ', '.join(map(str, answers))
prompt = agent_a.build_chat_prompt(new_input)
output = agent_a.ollama_chat(prompt=prompt, temperature=0.1)
print("FINAL ANSWER: ", output)

quit_ollama(model)
