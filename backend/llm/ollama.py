import requests


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen3:8b",
        url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.url = url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()


ollama_llm = OllamaLLM()