import ollama

response = ollama.chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "user",
            "content": "Responda apenas com: OK"
        }
    ]
)

print(response["message"]["content"])