import httpx
import ollama
import pytest

from doc_extractor.llm import LLMUnavailableError, OllamaClient

MSGS = [{"role": "user", "content": "x"}]


def patch_client(monkeypatch, chat):
    seen = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            seen["init"] = kwargs

        def chat(self, **kwargs):
            seen["chat"] = kwargs
            return chat(**kwargs)

    monkeypatch.setattr(ollama, "Client", FakeClient)
    return seen


def test_request_options_and_timeout(monkeypatch):
    seen = patch_client(
        monkeypatch, lambda **kw: {"message": {"content": "{}"}}
    )
    assert OllamaClient("m").complete(MSGS) == "{}"
    assert seen["init"]["timeout"] == 300
    assert seen["chat"]["options"] == {"temperature": 0, "num_ctx": 8192}
    assert seen["chat"]["think"] is False


@pytest.mark.parametrize("exc", [
    ollama.ResponseError("model 'm' not found", 404),
    ollama.ResponseError("boom", 500),
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("refused"),
    ConnectionError("refused"),
])
def test_errors_map_to_llm_unavailable(monkeypatch, exc):
    def chat(**kw):
        raise exc

    patch_client(monkeypatch, chat)
    with pytest.raises(LLMUnavailableError) as info:
        OllamaClient("m").complete(MSGS)
    assert str(info.value)
