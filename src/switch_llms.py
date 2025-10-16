from litellm import completion
from dotenv import load_dotenv
import os

#Available Local Models(via Llama)
llama_model="ollama/llama3.2"
mistral_model = "ollama/mistral:7b"
gemini_model = "gemini/gemini-2.5-flash" #Free Tier https://aistudio.google.com/
gpt_model = "gpt-4o"

# Required Env vars
load_dotenv() # Ensure .env with API Keys is loaded
llm_env_vars = ["GEMINI_API_KEY", "OPENAI_API_KEY"]

#Enable Paid LLMS
use_proprietary_llms = False
use_llama_llms = True


def ask_llm(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = completion(model=model, messages=messages)
    return response['choices'][0]['message']['content']


def api_keys_exist(env_vars, debug=False):

    for env in env_vars:
        value = os.environ.get(env)
        if value is None:
            return False
    return True


if use_proprietary_llms:
    if api_keys_exist(env_vars=llm_env_vars, debug=True):
        print(ask_llm(prompt="Who are you?", model=gpt_model))
        print(ask_llm(prompt="Who are you?", model=gemini_model))
    else:
        print("Ask for .env file from paACode!!!")
if use_llama_llms:
    print(ask_llm(prompt="Who are you?", model=llama_model))
    print(ask_llm(prompt="Who are you?", model=mistral_model))

#Todo: Maybe needed?
'''
from litellm.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
'''




