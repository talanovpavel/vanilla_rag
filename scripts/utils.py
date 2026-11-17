import csv
from pathlib import Path
import yaml

import json

def log_results(results_path,
                name: str,
                row: dict):
    """
    Write one row of experiment to CSV file
    """
    row['name'] = name

    results_path = Path(results_path)

    file_exists = results_path.is_file()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def load_config(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_prompts(questions_path: str,
                 system: str = ("Briefly answer the following question. Use Russian language. "
                                "Answer in plain text, without markdown, tables or lists.")
                 ) -> list[list[dict]]:
    """

    :param questions_path: path to json file with questions
    :return: chat template for LLM query
    """

    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    messages = []

    for i in data:
        messages.append([
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": i['question']
            }]
        )

    return messages