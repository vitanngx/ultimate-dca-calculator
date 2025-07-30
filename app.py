# app.py

import streamlit as st
from dca_calculator2 import calculate_dca  # Giả sử bạn có hàm chính tên là calculate_dca

st.title("DCA Calculator")

amount = st.number_input("Investment amount ($)", min_value=1)
price = st.number_input("Asset price ($)", min_value=0.01)
frequency = st.selectbox("Investment frequency", ["Daily", "Weekly", "Monthly"])

if st.button("Calculate"):
    result = calculate_dca(amount, price, frequency)
    st.write("Result:", result)
