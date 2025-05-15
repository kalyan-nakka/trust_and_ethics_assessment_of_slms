# version update
# https://github.com/openai/openai-python/discussions/742
import time
import os

from openai import OpenAI, OpenAIError

API_KEY = "openai_api_key"
client = OpenAI(api_key=API_KEY)

def gpt_single_try(messages, model = "gpt-3.5-turbo-0613"):
    response = client.chat.completions.create(
        model=model,
        messages = messages)

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def gpt(messages, model = "gpt-3.5-turbo-0613", num_retries=3):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_try(messages, model)
            break
        except OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(6)
    return r


def gpt_single_easy_try(user_input, model="gpt-3.5-turbo-0613",
                        system_role = "You are a helpful assistant."):
    response = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_input},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def gpt_easy(user_input, model="gpt-3.5-turbo-0613", 
             system_role="You are a helpful assistant.", num_retries=3):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_easy_try(user_input, model=model, system_role=system_role)
            break
        except OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)
    return r