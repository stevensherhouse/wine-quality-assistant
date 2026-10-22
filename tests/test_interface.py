import json
from types import SimpleNamespace
import pytest
from src.interface import EXAMPLE, InterfaceError, WineAssistant, WineLLM, validate_features

class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])

def provider(content):
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return WineLLM(client=client, model='test-model'), completions

def test_parser_extracts_provider_values(measurements):
    # No external API call: validates the real prompt/response adapter with a fixed provider response.
    llm, calls = provider(json.dumps({'features': measurements, 'ambiguous': [], 'in_scope': True}))
    parsed = llm.parse(EXAMPLE)
    assert parsed['features'] == measurements
    assert calls.calls[0]['messages'][1]['content'] == EXAMPLE
    assert calls.calls[0]['response_format']['type'] == 'json_schema'
    assert set(calls.calls[0]['response_format']['json_schema']['schema']['properties']['features']['required']) == set(measurements)
    assert 'Never invent' in calls.calls[0]['messages'][0]['content']

def test_null_is_missing_but_zero_is_preserved():
    llm, _ = provider(json.dumps({'features': {'citric_acid': 0, 'alcohol': None},
                                 'ambiguous': [], 'in_scope': True}))
    assert llm.parse('citric acid is zero')['features'] == {'citric_acid': 0}

@pytest.mark.parametrize('content', ['not json', '[]', '{"features": []}',
    '{"features":{"invented":1},"ambiguous":[],"in_scope":true}'])
def test_malformed_provider_response(content):
    llm, _ = provider(content)
    with pytest.raises(InterfaceError):
        llm.parse('please rate my wine')

def assistant_with(features, ambiguous=None, in_scope=True):
    llm, _ = provider(json.dumps({'features': features, 'ambiguous': ambiguous or [], 'in_scope': in_scope}))
    return WineAssistant(llm)

def test_incomplete_input_never_invokes_model(monkeypatch):
    assistant = assistant_with({'wine_type': 'red', 'alcohol': 11})
    monkeypatch.setattr(assistant.model, 'predict', lambda _: pytest.fail('Incomplete input reached model'))
    result = assistant.respond('red wine, 11% alcohol')
    assert result['status'] == 'clarify'
    assert 'fixed acidity' in result['message']

def test_out_of_scope_never_invokes_model(monkeypatch):
    assistant = assistant_with({}, in_scope=False)
    monkeypatch.setattr(assistant.model, 'predict', lambda _: pytest.fail('Out-of-scope input reached model'))
    assert assistant.respond('What is this wine worth?')['status'] == 'out_of_scope'

def test_ambiguous_acidity_never_invokes_model(monkeypatch):
    assistant = assistant_with({'wine_type': 'red'}, ['Is acidity fixed or volatile?'])
    monkeypatch.setattr(assistant.model, 'predict', lambda _: pytest.fail('Ambiguous input reached model'))
    assert assistant.respond('red wine, acidity 7.4')['status'] == 'clarify'

def test_out_of_range_and_invalid_values(trained, measurements):
    _, _, metadata = trained
    for value in [float('nan'), float('inf'), True, '11', 99]:
        assert validate_features({**measurements, 'alcohol': value}, metadata['bounds'])[1]

def test_multiturn_invokes_actual_model(trained, measurements):
    assistant = assistant_with({'alcohol': measurements['alcohol']})
    partial = {k: v for k, v in measurements.items() if k != 'alcohol'}
    assistant.llm.explain = lambda payload: f"Estimated score {payload['predicted_score']}/10."
    result = assistant.respond('alcohol 9.4%', previous=partial)
    assert result['status'] == 'prediction'
    assert 0 <= result['score'] <= 10
    assert result['features'] == measurements

def test_explanation_outage_preserves_real_prediction(measurements):
    assistant = assistant_with(measurements)
    def unavailable(_):
        raise InterfaceError('unavailable')
    assistant.llm.explain = unavailable
    result = assistant.respond(EXAMPLE)
    assert result['status'] == 'prediction'
    assert result['warning']
    assert f"{result['score']:.2f}" in result['message']
