import json
import os

import openai
from together import Together

from openai import OpenAI

from config import *
from constants.constants import *


def create_messages_for_query(prompt: str, feedback_message=None, assistant_message=None):
    messages_list = []

    prompt_message = {"role": "user", "content": prompt}
    messages_list.append(prompt_message)

    if assistant_message is not None:
        messages_list.append({"role": "assistant", "content": assistant_message})

    if feedback_message is not None:
        messages_list.append({"role": "user", "content": feedback_message})

    return messages_list


def query_openai_model(messages_list):
    openai.api_key = OPENAI_API_KEY

    is_query_failed = False

    should_try_query = True
    attempt = 1

    llm_raw_response = ""

    while should_try_query and attempt <= MAX_ATTEMPTS_FOR_API_QUERY:
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL_VERSION,
                messages=messages_list,
                seed=42,
            )

            llm_raw_response = response.choices[0].message.content

            should_try_query = False
            is_query_failed = False

        except Exception as e:
            print(f"Exception while using open.ai. Error: {e}")
            is_query_failed = True
            attempt += 1

    return llm_raw_response, is_query_failed


def query_togetherai_model(messages_list):
    os.environ["TOGETHER_API_KEY"] = TOGETHERAI_API_KEY

    is_query_failed = False

    should_try_query = True
    attempt = 1

    llm_raw_response = ""

    client = Together()

    while should_try_query and attempt <= MAX_ATTEMPTS_FOR_API_QUERY:
        try:
            response = client.chat.completions.create(
                model=TOGETHERAI_MODEL_VERSION,
                messages=messages_list
            )

            llm_raw_response = response.choices[0].message.content

            should_try_query = False
            is_query_failed = False

        except Exception as e:
            print(f"Exception while using open.ai. Error: {e}, Attempt {str(attempt)}")
            is_query_failed = True
            attempt += 1

    return llm_raw_response, is_query_failed


def query_openrouter_model(messages_list):
    is_query_failed = False
    should_try_query = True
    attempt = 1
    llm_raw_response = ""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    while should_try_query and attempt <= MAX_ATTEMPTS_FOR_API_QUERY:
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL_VERSION,
                messages=messages_list,
            )

            llm_raw_response = response.choices[0].message.content

            should_try_query = False
            is_query_failed = False

        except Exception as e:
            print(f"Exception while using OpenRouter. Error: {e}, Attempt {str(attempt)}")
            is_query_failed = True
            attempt += 1

    return llm_raw_response, is_query_failed


def safe_str_to_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"JSON parsing error: {e}")
        return {}


def create_accumulative_mapping_for_instance(ranked_sentences_list):
    accumulative_mapping = {}
    accumulative_sentences_list = []

    for idx, sentence in enumerate(ranked_sentences_list, start=1):
        accumulative_sentences_list.append(sentence)
        accumulative_mapping[idx] = {SELECTED_SNIPPETS: accumulative_sentences_list.copy()}

    return accumulative_mapping

