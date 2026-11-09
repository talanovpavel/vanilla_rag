import logging

import json

from vllm import SamplingParams
from vllm import LLM

import click

from utils import load_config

import os
import pandas as pd


def make_prompts(questions_path: str,
                 system: str = "Briefely answer the following question. Use Russian language"
                 ) -> list[list[dict]]:
    """

    :param questions_path: path to json file with questions
    :return: chat template for vllm inference
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


def get_outputs(messages: list[list[dict]],
                llm_params: dict,
                sampling_params: dict
                ) -> list:
    """
    :param messages:
    :param llm_params:
    :param sampling_params:
    :return:
    """

    llm = LLM(**llm_params)

    outputs = llm.chat(
        messages,
        SamplingParams(**sampling_params)
    )

    return [o.outputs[0].text for o in outputs]


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    config = load_config(config_path)

    questions_path = config["data"]["questions_path"]
    output_path = config["data"]["output_path"]
    llm_params = config["llm"]
    sampling_params = config["sampling"]
    result_path = config["result_path"]

    messages = make_prompts(questions_path)
    answers = get_outputs(messages, llm_params, sampling_params)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    df = pd.DataFrame(questions)
    df["answer"] = answers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
