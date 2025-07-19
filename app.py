import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sea 

st.title("📊 Sample Dashboard")

file = st.file_uploader("UPLOAD YOUR FILE", type="csv")

if file:
    df = pd.read_csv(file)
    st.write("Data Preview")
    st.dataframe(df)
    st.write("Summary")
    st.write(df.describe())
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        sea.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    s =df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots()
    sea.scatterplot(s,ax=ax)
    st.pyplot(fig)