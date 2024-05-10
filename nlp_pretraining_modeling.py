# LLM(거대 생성형 언어 모델 (Attention 등 고성능 모델 이용))
# OPENAI(GPT-3.5-turbo)(GPT3.5)
from openai import OpenAI

client = OpenAI(api_key='open')

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "무엇을 도와드릴까요?"},
    {"role": "user", "content": "khuthon을 잘하려면 어떻게 해야 돼?"}
  ]
)

print(completion.choices[0].message)

# OPENAI Embedding(text-embedding-3-small)
from openai import OpenAI
client = OpenAI(api_key='open')

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)

print(response.data[0].embedding)