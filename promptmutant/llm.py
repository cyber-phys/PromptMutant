import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_chat(prompt, model="gpt-3.5-turbo"):
    system="You are a helpful assistant."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content":  prompt},
        ]
    )
    return completion.choices[0].message["content"]

def openai_instruct(prompt, model="gpt-3.5-turbo-instruct"):
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].text


