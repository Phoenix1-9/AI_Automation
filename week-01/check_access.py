import os
import time  
import json 
import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "env_key": "GEMINI_API_KEY",
        "headers": lambda key: {"x-goog-api-key": key},
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "env_key": "GROQ_API_KEY",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/models",
        "env_key": "OPENROUTER_API_KEY",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
    },
}


def check_provider(name: str, config: dict) -> dict:
    key = os.environ.get(config["env_key"])
    if not key:
        print(f"[{name}] کلید تنظیم نشده ({config['env_key']})")
        return {
            "provider": name,
            "status_code": None,
            "response_time_sec": None,
            "error": "API key not set",
        }

    start_time = time.perf_counter()

    try:
        response = requests.get(
            config["url"],
            headers=config["headers"](key),
            timeout=15,
        )
        elapsed_time = round(time.perf_counter() - start_time, 3)
    except requests.exceptions.RequestException as error:
        elapsed_time = round(time.perf_counter() - start_time, 3)
        print(f"[{name}] خطای شبکه: {type(error).__name__} ({elapsed_time}s)")
        return {
            "provider": name,
            "status_code": None,
            "response_time_sec": elapsed_time,
            "error": type(error).__name__,
        }

    print(f"[{name}] Status: {response.status_code} | Time: {elapsed_time}s")
    if response.status_code != 200:
        print(f"    پاسخ: {response.text[:200]}")

    return {
        "provider": name,
        "status_code": response.status_code,
        "response_time_sec": elapsed_time,
    }


if __name__ == "__main__":
    results = []
    for provider_name, provider_config in PROVIDERS.items():
        result = check_provider(provider_name, provider_config) 
        results.append(result)

    output_file = "access_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\n✅ نتایج با موفقیت در فایل {output_file} ذخیره شد.")
