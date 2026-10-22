from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.demo import RecordedExtraction
from src.interface import EXAMPLE, InterfaceError

def test_app_setup_message_without_credentials():
    with patch('src.interface.WineLLM', side_effect=InterfaceError('Configure Nebius credentials.')):
        app = AppTest.from_file('src/app.py').run(timeout=20)
    assert not app.exception
    assert 'Configure Nebius' in app.info[0].value

def test_app_complete_and_incomplete_queries():
    with patch('src.interface.WineLLM', return_value=RecordedExtraction()):
        app = AppTest.from_file('src/app.py').run(timeout=20)
        app.chat_input[0].set_value(EXAMPLE).run(timeout=20)
        assert not app.exception
        assert app.metric[0].label == 'Estimated quality'
        assert app.session_state['features']['wine_type'] == 'red'
        app.button[0].click().run(timeout=20)
        app.chat_input[0].set_value('I have a red wine with 11% alcohol. Rate it.').run(timeout=20)
        assert not app.exception
        assert not app.metric
        assert 'Please provide:' in app.session_state['messages'][-1]['content']
