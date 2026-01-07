def fetch_questions(file_path: str) -> list[str]:
    questions_list = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                questions_list.append(line.strip())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return questions_list

# def push_answers(answers, file_path: str):
#     try:
#         with open(file_path, 'w') as file:
#             cnt = 1
#             for response in answers:
#                 file.write(f"{cnt}. {response}\n")
#                 cnt += 1
#     except Exception as e:
#         print("error in pushing responses into file")


def push_answers(answers, file_path: str):
    """
    Write answers to file.
    
    Args:
        answers: List of answer strings
        file_path: Path to output file
        append: If True, append to existing file. If False, overwrite.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for idx, response in enumerate(answers, start=1):
                file.write(f"{idx}. {response}\n\n")
    except Exception as e:
        print(f"Error in pushing responses into file: {e}")



