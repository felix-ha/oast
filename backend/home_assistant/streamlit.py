import streamlit as st
from stt import get_text
from llm import call_llm
from tts import get_speech

audio_value = st.audio_input("Stelle eine Frage")

if audio_value:
    recognized_text = get_text(audio_value) 
    st.markdown(f"""
    ### Erkannter Text
    """)
    st.write(recognized_text)

    # recognized_text = "Wie viel Buchstaben E enthält das Wort Erdbeere? Denke schritt für schritt."
    answer = call_llm(recognized_text)
    st.markdown(f"""
    ### Antwort von LLAMA
    """)
    st.write(answer)

    path = "data/streamlit.wav"
    get_speech(answer, path)
    st.audio(path)



# --find-links https://download.pytorch.org/whl/cu124
# from streamlit_mic_recorder import mic_recorder
# audio_value = mic_recorder(
#     start_prompt="Start recording",
#     stop_prompt="Stop recording",
#     just_once=False,
#     use_container_width=False,
#     callback=None,
#     args=(),
#     kwargs={},
#     key=None
# )
# if audio_value:
#     st.audio(audio_value['bytes'])

