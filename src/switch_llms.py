from litellm import completion

#Available Local Models(via Llama)
llama_model="ollama/llama3.2"
mistral_model = "ollama/mistral:7b"


def ask_llm(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = completion(model=model, messages=messages)
    return response['choices'][0]['message']['content']

print(ask_llm(prompt="Who are you?", model= llama_model))
print(ask_llm(prompt="Who are you?", model= mistral_model))



