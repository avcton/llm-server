import openai

openai.api_key = "dummy" 
openai.base_url = "https://4ipkhe13jpcjz1-8888.proxy.runpod.net/v1/"

stream = openai.chat.completions.create(
    model="ALLaM-7B-Instruct-preview",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer only in Arabic."},
        {"role": "user", "content": "I'm doing well, thank you! Can you tell me about you?"},
        {"role": "assistant", "content": "مرحبًا! أنا بخير، شكرًا لسؤالك. كيف حالك؟"},
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

print("\nDone!")
