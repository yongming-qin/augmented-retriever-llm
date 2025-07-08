"""
A unified interface for calling different LLM providers.

"""
import os
from openai import OpenAI
from together import Together
from anthropic import Anthropic
import dotenv

dotenv.load_dotenv()


class LmClient:
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        print(f"Initializing {model} with temperature {temperature}")
        
        self.is_together = "deepseek" in model or "llama" in model or "Qwen" in model or "mistral" in model
        self.is_openai = "gpt" in model
        self.is_claude = "claude" in model
        
        if self.is_openai:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        elif self.is_together:
            from together import Together
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif self.is_claude:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError("Unsupported provider")

    def chat(self, messages: list):
        if self.is_openai:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            ).choices[0].message.content

        elif self.is_together:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            ).choices[0].message.content

        elif self.is_claude:
            return self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=self.temperature
            ).content[0].text
        else:
            raise ValueError("Unsupported model")

"""
together:
    model = "deepseek-ai/DeepSeek-R1-0528-tput"
    model = "deepseek-ai/DeepSeek-V3"
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    
claude:
    model = "claude-3-5-haiku-20241022"
    model = "claude-3-5-sonnet-20240620"
"""

if __name__ == "__main__":
    llm = LmClient(model="claude-3-5-haiku-20241022")
    response = llm.chat(messages=[{"role": "user", "content": "Hi, how are you?"}])
    print(response)