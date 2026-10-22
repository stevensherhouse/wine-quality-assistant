"""Run from repository root: streamlit run src/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.interface import EXAMPLE, InterfaceError, WineAssistant, WineLLM
from src.preprocess import UNITS

st.set_page_config(page_title='Wine Quality Assistant', page_icon='🍷', layout='centered')
st.title('Wine Quality Assistant')
st.write('Turn a laboratory wine analysis into an estimated quality score. Ask in plain English, then add any missing measurements.')
st.caption('For red and white Portuguese Vinho Verde wines. Scores estimate sensory ratings from 0 to 10.')
with st.sidebar:
    st.header('Start here')
    st.write('1. Supply red or white wine type and your laboratory measurements.\n2. Answer any clarification requests.\n3. Review the estimate and model limitations.')
    if st.button('Start a new wine'):
        st.session_state.features = {}
        st.session_state.messages = []
        st.rerun()
    with st.expander('Required measurements and units'):
        st.json(UNITS)
    st.caption('Default mode runs Qwen locally on this computer with no paid API. Optional Nebius mode sends measurements to Nebius.')
with st.expander('Example question'):
    st.code(EXAMPLE, language=None, wrap_lines=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.features = {}
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
try:
    assistant = WineAssistant(WineLLM())
except InterfaceError as error:
    st.info(str(error))
    st.caption('See README.md for local model setup. Optional provider keys belong in .env, never in a chat message.')
    st.stop()
query = st.chat_input('Describe your wine and its measurements')
if query:
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Reading the measurements…'):
                result = assistant.respond(query, st.session_state.features)
            st.session_state.features = result['features']
            if result['status'] == 'prediction':
                st.metric('Estimated quality', f'{result["score"]:.2f} / 10')
                st.caption(f'Test MAE: {assistant.metadata["test_metrics"]["mae"]:.3f} points. This is average error, not an individual prediction interval.')
                if result['warning']:
                    st.warning(result['warning'])
            st.write(result['message'])
            with st.expander('Measurements used'):
                st.json(result['features'])
            content = result['message']
            if result['status'] == 'prediction':
                content = f'Estimated quality: {result["score"]:.2f}/10\n\n' + content
            st.session_state.messages.append({'role': 'assistant', 'content': content})
        except InterfaceError as error:
            st.error(str(error))
