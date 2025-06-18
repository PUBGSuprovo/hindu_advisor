import requests

def test_groq_with_key():
    api_key = "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"
    url = "https://api.groq.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        print("Groq API response:", data['choices'][0]['message']['content'])
    except Exception as e:
        print("API call failed:", e)

if __name__ == "__main__":
    test_groq_with_key()
