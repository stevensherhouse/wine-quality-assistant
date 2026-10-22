"""Opt-in real LLM integration check, local by default; ordinary CI uses doubles."""
import json
import math
from datetime import datetime, timezone
from src.interface import EXAMPLE, WineAssistant, WineLLM, ROOT

def main():
    assistant = WineAssistant(WineLLM())
    complete = assistant.respond(EXAMPLE)
    assert complete['status'] == 'prediction', complete
    assert complete['warning'] is None, 'Live explanation failed'
    assert complete['features']['fixed_acidity'] == 7.4
    assert complete['features']['alcohol'] == 9.4
    assert complete['features']['wine_type'] == 'red'
    expected = dict(zip(
        ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
         'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
         'pH', 'sulphates', 'alcohol'],
        [7.4, .70, 0, 1.9, .076, 11, 34, .9978, 3.51, .56, 9.4]))
    for name, value in expected.items():
        assert math.isclose(complete['features'][name], value, rel_tol=1e-8), (name, complete)
    incomplete = assistant.respond('I have a red wine with 11% alcohol. Rate it.')
    assert incomplete['status'] == 'clarify', incomplete
    ambiguous = assistant.respond('Rate a red wine with acidity 7.4.')
    assert ambiguous['status'] == 'clarify', ambiguous
    out_of_scope = assistant.respond('What is the price of Bitcoin?')
    assert out_of_scope['status'] == 'out_of_scope', out_of_scope
    followup = assistant.respond(
        'Here are the remaining wine measurements: fixed acidity 7.4 g/L, volatile acidity 0.70 g/L, '
        'citric acid 0 g/L, residual sugar 1.9 g/L, chlorides 0.076 g/L, free sulfur dioxide 11 mg/L, '
        'total sulfur dioxide 34 mg/L, density 0.9978 g/cm3, pH 3.51, sulphates 0.56 g/L.',
        previous=incomplete['features'])
    assert followup['status'] == 'prediction', followup
    assert followup['features']['alcohol'] == 11
    white = assistant.respond(
        'Can you estimate the quality of my white wine? Its fixed acidity is 7.0 g/L and '
        'volatile acidity is 0.27 g/L. Citric acid is 0.36 g/L; residual sugar is 20.7 g/L; '
        'chlorides are 0.045 g/L. Free sulfur dioxide is 45 mg/L and total sulfur dioxide '
        'is 170 mg/L. Density is 1.001 g/cm3, pH is 3.00, sulphates are 0.45 g/L, '
        'and alcohol is 8.8% by volume.')
    assert white['status'] == 'prediction', white
    assert white['features']['wine_type'] == 'white'
    assert white['features']['residual_sugar'] == 20.7
    report = {'mode': 'live_' + assistant.llm.provider, 'llm_model': assistant.llm.model,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'complete': complete, 'incomplete': incomplete,
              'ambiguous': ambiguous, 'out_of_scope': out_of_scope,
              'followup': followup, 'white_wine': white}
    (ROOT / 'reports/live_check.json').write_text(json.dumps(report, indent=2))
    print(f'Live {assistant.llm.provider} extraction, prediction, explanation, and edge-case checks passed.')

if __name__ == '__main__':
    main()
