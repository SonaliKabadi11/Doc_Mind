import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.exception import LLMError, LLMTimeoutError
from app.generation.openai_compatible import OpenAICompatibleProvider


class StubClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def provider():
    return OpenAICompatibleProvider(
        model="m", base_url="http://localhost:9", api_key=SecretStr("dummy")
    )


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://localhost:9/chat/completions")


def test_returns_stripped_text(provider):
    provider._client = StubClient(result=AIMessage(content="  hello  "))
    assert provider.generate("sys", "user") == "hello"


def test_list_content_is_joined(provider):
    blocks = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    provider._client = StubClient(result=AIMessage(content=blocks))
    assert provider.generate("s", "u") == "hello"


def test_timeout_maps_to_llm_timeout_error(provider):
    provider._client = StubClient(error=openai.APITimeoutError(request=_req()))
    with pytest.raises(LLMTimeoutError):
        provider.generate("s", "u")


def test_api_error_maps_to_llm_error(provider):
    err = openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_req()), body=None
    )
    provider._client = StubClient(error=err)
    with pytest.raises(LLMError) as exc:
        provider.generate("s", "u")
    assert not isinstance(exc.value, LLMTimeoutError)


def test_empty_response_raises(provider):
    msg = AIMessage(content="", response_metadata={"finish_reason": "length"})
    provider._client = StubClient(result=msg)
    with pytest.raises(LLMError, match="length"):
        provider.generate("s", "u")


def test_missing_api_key_fails_fast():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model="m", base_url="http://x", api_key=None)