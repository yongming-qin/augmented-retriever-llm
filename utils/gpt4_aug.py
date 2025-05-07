from openai import AzureOpenAI
import time
import pandas as pd
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


#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = "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"

client = AzureOpenAI(
            api_key="xxx",
            # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
            azure_endpoint=OPENAI_ENDPOINT,
            azure_deployment=OPENAI_DEPLOYMENT_MODEL,
            api_version=OPENAI_AZURE_API_VERSION,
            http_client=httpx.Client(verify=False),
            default_headers={
                'Authorization': f'Bearer {get_llm_access_token()}',
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
            })
def request_response_from_gpt(prompt):
    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT_MODEL,
        #prompt=prompt,
        messages=[
            {"role": "system", "content": "You are a crowdsourcing worker that earns a living through creating paraphrases."},
            {"role": "user", "content": prompt}],
        temperature=1,
        frequency_penalty=0.0,
        presence_penalty=1.5,
        n=1)
    output = response.choices[0].message.content
    return output
def request_with_checks(prompt):
    success = False
    count = 0
    while not success:
        if count > 0:
            print(f'Retrying with again. Current number of retries: {count}')
        if count >= 10:
            raise Exception('Too many attempts')
        try:
            response = request_response_from_gpt(prompt)
            sucess = True
            break
        except openai.error.RateLimitError as e:
            print(e)
            time.sleep(10)
            count += 1
        except openai.error.APIConnectionError as e:
            print(e)
            time.sleep(5)
            count += 1
        except openai.error.APIError or openai.error.JSONDecodeError as e:
            print(e)
            time.sleep(5)
            count += 1
        except openai.error.Timeout as e:
            print(e)
            time.sleep(5)
            count += 1
        except openai.error.ServiceUnavailableError as e:
            print(e)
            time.sleep(5)
            count += 1
    return response

def collect_samples(dct_final_prompts, time_sleep: int=1):
    dct_responses = {}
    for idx, key in enumerate(dct_final_prompts):
        print("Now on label no. {} out of {}.".format(idx, len(dct_final_prompts.keys())))
        dct_responses[key] = []
        for prompt in dct_final_prompts[key]:
            #print("prompt: {}".format(prompt))
            response = request_response_from_gpt(prompt)
            #print("response: {}".format(response))
            #exit(0)
            dct_responses[key].append(response)
            time.sleep(time_sleep)
    return dct_responses


import re
import string


def filter_responses(dct_responses):
    dct_df = {'label': [], 'text': []}
    for key in dct_responses:
        for responses in dct_responses[key]:
            #print(responses)
            #print("responses[0]", responses[0])
            #print("responses[1]", responses[1])
            contents = responses.split('Paraphrase')
            # for response in responses[0].choices:
            #     contents = response.message.content.split('\n')
            for content in contents:
                if len(content) == 0:
                    continue
                if content != '':
                    content = content[2:]
                    #print("content", content)
                    dct_df['label'].append(key)
                    dct_df['text'].append(content)
            #exit(0)
                    #dct_df['seed'].append(responses[1])

    fb_0 = pd.DataFrame.from_dict(dct_df)

    #fb_0['text'] = fb_0['text'].apply(lambda x: x.lower())
    fb_0['text'] = fb_0['text'].apply(lambda x: x.replace('"', ''))
    fb_0['text'] = fb_0['text'].apply(lambda x: x.replace('\n', ''))
    fb_0['text'] = fb_0['text'].apply(lambda x: x.replace(':', ''))

    #fb_0['text'] = fb_0['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

    return fb_0

