import streamlit as st
import requests

st.title("Job Candidate Recommender")

job_id = st.number_input("Enter Job ID", min_value=1)
top_k = st.slider("Top K", 1, 10, 5)

if st.button("Recommend"):
    response = requests.get(
        f"http://localhost:8000/recommend/{job_id}?top_k={top_k}"
    )
    st.write(response.json())