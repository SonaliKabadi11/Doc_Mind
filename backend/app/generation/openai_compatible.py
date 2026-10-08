import logging

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.exception import LLMError, LLMTimeoutError
from app.generation.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible chat endpoint (Gemini, OpenAI, Ollama, ...)."""

    def __init__(self,
                 *, 
                 model:str, 
                 base_url:str, 
                 api_key:SecretStr|None, 
                 temperature: float = 0.0 , 
                 max_tokens:int = 1024, 
                 timeout: float=30.0, 
                 max_retries : int=2) -> None:
        if api_key is None:
            raise ValueError("LLM API key is not configured (set GOOGLE_API_KEY)")
        self.model = model
        self._client = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,  # SDK retries 429/5xx/timeouts with backoff
        
        )
    def generate(self, system:str, user:str) -> str:
        messages = [SystemMessage(content=system),
                    HumanMessage(content=user)]
        try:
            response = self._client.invoke(messages)
        except openai.APITimeoutError as e:# must come before APIError (subclass)
            logger.exception("LLM call timed out (model=%s)", self.model)
            raise LLMTimeoutError("The language model took too long to respond") from e
        except openai.APIError as e:
            logger.exception("LLM call failed (model=%s)", self.model)
            raise LLMError(f"Language model request failed ({type(e).__name__})") from e

        text = _extract_text(response.content)
        if not text:
            finish = response.response_metadata.get("finish_reason")
            logger.error("LLM returned empty text (finish_reason=%s)", finish)
            raise LLMError(f"Language model returned an empty response (finish_reason={finish})")
        return text

    
def _extract_text(content:str | list) -> str:
    """LangChain content is a str, or a list of blocks for some providers."""
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts).strip()    


