import json
import os

def load_questions():
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'questions_math.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print("[ERROR] Specified question bank was not found.")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("[ERROR] Failed to decode the json file")


def choose_question(question_bank):
    while True:
        try:
            user_input = int(input("\n> "))
            if 1 <= user_input <= len(question_bank):
                selected = question_bank[user_input - 1]
                return selected
            else:
                print("[ERROR] Chosen number is out of range")
        except ValueError:
            print("[ERROR] Chosen value is not a valid number")


def display_questions(question_bank):
    print("Choose a question by typing in the corresponding number:\n")

    for number, question in enumerate(question_bank, start=1):
        print(f"{number}. {question['question']}")



def get_chosen_question():
    chosen_question = ""
    questions = load_questions()

    if questions:
        display_questions(questions)
        chosen_question = choose_question(questions)["question"]

    return chosen_question
