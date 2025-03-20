import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "Write a limerick about the Python Programming."},
    ],
)

response = completion.choices[0].message.content
print(response)
