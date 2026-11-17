import json
import logging
import time
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from scripts.utils import load_config, make_prompts

logger = logging.getLogger(__name__)


def get_responses(messages: list[list[dict]],
                  model_params: dict,
                  sampling_params: dict,
                  client: Groq | None = None,
                  max_retries: int = 5,
                  request_delay=2) -> list[str]:
    """Send each dialog to Groq and return answers in the same order."""
    client = client if client is not None else Groq()

    answers = []
    for i, m in enumerate(messages):
        for attempt in range(max_retries):
            try:
                time.sleep(request_delay)
                chat_completion = client.chat.completions.create(
                    messages=m,
                    **model_params,
                    **sampling_params,
                )
                answers.append(chat_completion.choices[0].message.content)
                break
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning("Rate limit on question %d, retry in %ds", i, wait)
                time.sleep(wait)
        else:
            logger.error("Question %d failed after %d retries", i, max_retries)
            answers.append(None)

        if (i + 1) % 10 == 0:
            logger.info("Done %d/%d", i + 1, len(messages))

    return answers


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    config = load_config(config_path)

    questions_path = config["data"]["questions_path"]
    output_path = Path(config["data"]["output_path"])
    model_params = config["llm"]
    sampling_params = config["sampling"]

    messages = make_prompts(questions_path)
    answers = get_responses(messages, model_params, sampling_params)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    df = pd.DataFrame(questions)
    df["answer"] = answers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()



















