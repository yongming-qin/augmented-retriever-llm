# from idam_token_generator import get_llm_access_token
import os

import httpx
from openai import AzureOpenAI

os.environ["APP_CLIENT_ID"] = "hQEwFjGeJuhAbybrxmHBPS3gNmsa"
os.environ["APP_CLIENT_SECRET"] = "FuFxdfg9fTw3K1Ifjcma85GABJMa"

from llm_idam_token_generator.idam_token_generator import get_llm_access_token

# OpenAI Endpoint details
OPENAI_ENDPOINT = "https://openai-llm-frontdoor-hma7evbthrd4cugn.a01.azurefd.net"
OPENAI_DEPLOYMENT_MODEL = "gpt-4-32k-beta"
OPENAI_AZURE_API_VERSION = "2023-12-01-preview"
APIM_KEY = "5b87b3b8-4743-4dfb-9c7b-e04e107dbdc0"


def summarise(text):
    # Setting up the OpenAI Client
    client = AzureOpenAI(
        api_key="xxx",  # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_DEPLOYMENT_MODEL,
        api_version=OPENAI_AZURE_API_VERSION,
        http_client=httpx.Client(verify=False),
        default_headers={
            'Authorization': f'Bearer {get_llm_access_token()}',
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
        })

    # Creating Chat Completion
    completion = client.chat.completions.create(model=OPENAI_DEPLOYMENT_MODEL, messages=text)
    response = completion.choices[0].message.content

    # Printing the response
    print(response)
    return response


summarise([
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": "what is 1 plus 1?"
    },
])