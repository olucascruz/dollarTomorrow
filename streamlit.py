import streamlit as st
import pandas as pd
import numpy as np
from bcb_service import get_currency
from model import get_next_day_currency

st.title("Cotação do dolar")
st.line_chart(get_currency(180)[-7:])

st.title("Previsão para amanhã:")
st.markdown(f"#### {get_next_day_currency()[0]}")
if get_next_day_currency()[1]:
    st.markdown("#### :green[Sobe]")
else:
    st.markdown("#### :red[Desce]")
