from types import SimpleNamespace
import pytest
from casp_mem.clients.llm_client import QwenClient

class FakeCompletions:
    def __init__(self, message): self.message = message
    def create(self, **kwargs): return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])

def make_client(message):
    obj=QwenClient.__new__(QwenClient)
    obj.model='qwen'
    obj.client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(message)))
    return obj

def test_thinking_review_rejects_server_ignoring_mode():
    client=make_client(SimpleNamespace(content='{"ok":true}', model_extra={}))
    with pytest.raises(ValueError, match='thinking'):
        client._json('instruction', {}, thinking=True)

def test_thinking_review_accepts_separate_reasoning_or_think_tags():
    client=make_client(SimpleNamespace(content='{"ok":true}', model_extra={'reasoning_content':'some reasoning'}))
    assert client._json('instruction', {}, thinking=True) == {'ok':True}
    client=make_client(SimpleNamespace(content='<think>some reasoning</think>{"ok":true}', model_extra={}))
    assert client._json('instruction', {}, thinking=True) == {'ok':True}

def test_non_thinking_request_rejects_thinking_output():
    client=make_client(SimpleNamespace(content='{"ok":true}', model_extra={'reasoning_content':'some reasoning'}))
    with pytest.raises(ValueError, match='non-thinking'):
        client._json('instruction', {}, thinking=False)
