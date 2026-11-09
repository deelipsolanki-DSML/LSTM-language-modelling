from prediction_engine import prediction
import streamlit as st
from st_keyup import st_keyup

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
  st.header("LSTM based - Next Word Predictor")

st.space("medium")
inp = st_keyup("Type here:", debounce=20)
# What does debounce=20 do?
# - Waits 20 milliseconds after your last keystroke.
# - Automatically updates the input value.
# - Triggers Streamlit to rerun prediction code.
# - No need to press Enter

if inp:
    word, prob = prediction(inp, top_k=5)
    st.write('suggestions')
    st.write(word)