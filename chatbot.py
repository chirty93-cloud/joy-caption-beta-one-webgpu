import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def tanya_ai(prompt_pengguna):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "https://openrouter.ai",
        "X-Title": "bintang-ai"
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt_pengguna}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

def jalankan_chatbot():
    print("🔌 AI Kelistrikan (Online - Bintang Edition) ⚡")
    print("Tanyakan apa saja tentang listrik. Ketik 'keluar' untuk berhenti.")
    while True:
        pertanyaan = input("Kamu: ")
        if pertanyaan.lower() == "keluar":
            break
        jawaban = tanya_ai(pertanyaan)
        print("AI:", jawaban)
