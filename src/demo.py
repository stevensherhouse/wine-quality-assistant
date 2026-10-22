"""Repeatable smoke demo; --live exercises the configured real language model."""
import argparse
import json
from src.interface import EXAMPLE, WineAssistant, WineLLM
from src.preprocess import FEATURES

class RecordedExtraction:
    """Explicit test double for an offline smoke demo; never used by the app."""
    def parse(self, query):
        if query == EXAMPLE:
            return {'features': dict(zip(FEATURES,
                [7.4, .70, 0, 1.9, .076, 11, 34, .9978, 3.51, .56, 9.4, 'red'])),
                'ambiguous': [], 'in_scope': True}
        if query == 'I have a red wine with 11% alcohol. Rate it.':
            return {'features': {'wine_type': 'red', 'alcohol': 11}, 'ambiguous': [], 'in_scope': True}
        return {'features': {}, 'ambiguous': [], 'in_scope': False}
    def explain(self, payload):
        return (f"Estimated quality: {payload['predicted_score']}/10. "
                f"Held-out MAE: {payload['test_mae']} points; not a confidence interval. "
                'Portuguese Vinho Verde training data; subjective taste and other wine styles may differ.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    print('LIVE LLM DEMO (configured local or Nebius model)' if args.live else 'OFFLINE SMOKE DEMO: recorded extraction; actual trained ML model; no live LLM')
    assistant = WineAssistant(WineLLM() if args.live else RecordedExtraction())
    for query in [EXAMPLE, 'I have a red wine with 11% alcohol. Rate it.', 'What is the price of Bitcoin?']:
        print('\nUSER:', query)
        print(json.dumps(assistant.respond(query), indent=2))
