import streamlit as st
import requests

API_URL = "http://localhost:8000/vector-memory"

st.title("RAG Chat with Documents")

mode = st.radio("Select input type", ["URL", "File"])

if mode == "URL":
    url = st.text_input("Enter URL")
    if st.button("Add URL"):
        r = requests.post(f"{API_URL}/documents/url", json={"url": url})
        st.write(r.json())
else:
    file = st.file_uploader("Upload File", type=["pdf","docx"])
    if file and st.button("Add File"):
        r = requests.post(f"{API_URL}/documents/file", files={"file": file})
        st.write(r.json())

query = st.text_input("Ask a question:")
if st.button("Search"):
    r = requests.get(f"{API_URL}/search", params={"query": query})
    results = r.json()
    for doc in results:
        st.write(doc["content"][:])
