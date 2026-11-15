from prediction_engine import prediction
import streamlit as st
from st_keyup import st_keyup

st.title("LSTM based Next Word Predictor", text_alignment = 'center')

st.space("medium")
inp = st_keyup("Type here:", debounce=20)
# What does debounce=20 do?
# - Waits 20 milliseconds after your last keystroke.
# - Automatically updates the input value.
# - Triggers Streamlit to rerun prediction code.
# - No need to press Enter

if inp.strip():
    words, prob = prediction(inp, top_k=5)
    if prob is not None:
        st.space("small")

        st.markdown(
            "<h3 style='font-size: 20px; font-weight: 600;'>suggestions</h3>",
            unsafe_allow_html=True,
        )

        pills_html = "".join(
        [
            f"<span style='background-color: #1e3a8a; color: #ffffff; padding: 6px 14px; margin-right: 8px; border-radius: 16px; font-size: 16px; font-family: sans-serif; border: 1px solid #464855;'>{word}</span>"
            for word in words
        ]
    )   
        st.markdown(
            f"<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;'>{pills_html}</div>",
            unsafe_allow_html=True,
        )