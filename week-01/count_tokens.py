import os
import requests 

from dotenv import load_dotenv
load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_KEY = os.environ.get("GEMINI_API_KEY")

TEXTS = {
    "english": "Please summarize this customer message and classify its priority.",
    "persian": "لطفاً این پیام مشتری را خلاصه کن و اولویت آن را مشخص کن.",
}


def pick_model(headers: dict) -> str:
    response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=15)
    response.raise_for_status()
    
    # کلماتی که نشان‌دهنده ناپایدار یا آزمایشی بودن مدل هستند
    bad_keywords = ["exp", "preview", "vision", "2.5"]
    
    for model in response.json().get("models", []):
        name = model["name"]
        methods = model.get("supportedGenerationMethods", [])
        
        if (
            "countTokens" in methods 
            and "flash" in name 
            and not any(bad in name for bad in bad_keywords)
        ):
            return name
            
    raise RuntimeError("مدل مناسبی پیدا نشد")


def count_tokens(model_name: str, text: str, headers: dict) -> int:
    response = requests.post(
        f"{BASE_URL}/{model_name}:countTokens",
        headers=headers,
        json={"contents": [{"parts": [{"text": text}]}]},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["totalTokens"]


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY تنظیم نشده")

    request_headers = {"x-goog-api-key": API_KEY}
    model = pick_model(request_headers)
    print(f"مدل: {model}")

    for label, text in TEXTS.items():
        tokens = count_tokens(model, text, request_headers)
        print(f"{label}: {len(text)} نویسه، {tokens} توکن")