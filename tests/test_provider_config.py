from unittest.mock import Mock
import pytest
import src.interface as interface

def setup(monkeypatch, provider):
    monkeypatch.setattr(interface, 'load_dotenv', lambda _: None)
    for name in ['LLM_MODEL', 'LLM_BASE_URL', 'NEBIUS_API_KEY', 'NEBIUS_MODEL', 'NEBIUS_BASE_URL']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LLM_PROVIDER', provider)
    factory = Mock()
    monkeypatch.setattr(interface, 'OpenAI', factory)
    return factory

def test_local_provider_needs_no_secret(monkeypatch):
    factory = setup(monkeypatch, 'local')
    llm = interface.WineLLM()
    assert llm.model == 'wine-local'
    assert llm.provider == 'local'
    assert factory.call_args.kwargs['base_url'] == 'http://127.0.0.1:8080/v1'

def test_local_provider_rejects_external_host(monkeypatch):
    factory = setup(monkeypatch, 'local')
    monkeypatch.setenv('LLM_BASE_URL', 'https://external.example/v1')
    with pytest.raises(interface.InterfaceError, match='loopback'):
        interface.WineLLM()
    factory.assert_not_called()

def test_nebius_requires_env_secret(monkeypatch):
    factory = setup(monkeypatch, 'nebius')
    with pytest.raises(interface.InterfaceError, match='NEBIUS_API_KEY'):
        interface.WineLLM()
    factory.assert_not_called()

def test_nebius_reads_secret_from_environment(monkeypatch):
    factory = setup(monkeypatch, 'nebius')
    monkeypatch.setenv('NEBIUS_API_KEY', 'test-only-fake-key')
    monkeypatch.setenv('NEBIUS_MODEL', 'test-chat-model')
    llm = interface.WineLLM()
    assert llm.model == 'test-chat-model'
    assert factory.call_args.kwargs['api_key'] == 'test-only-fake-key'

def test_unknown_provider_is_rejected(monkeypatch):
    factory = setup(monkeypatch, 'unsupported')
    with pytest.raises(interface.InterfaceError, match='LLM_PROVIDER'):
        interface.WineLLM()
    factory.assert_not_called()
