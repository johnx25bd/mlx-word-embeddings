import streamlit as st
import requests
import json

# FastAPI endpoint
API_URL = "http://localhost:8000"

st.title("Article Upload and Prediction")

# Form for manual input
st.header("Manual Input")
with st.form("manual_input"):
    title = st.text_input("Title")
    author = st.text_input("Author")
    url = st.text_input("URL")
    date = st.date_input("Date")
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        data = {"title": title, "author": author, "url": url, "date": str(date)}
        response = requests.post(f"{API_URL}/upload", data=data)
        if response.status_code == 200:
            result = response.json()
            st.success("Article submitted successfully!")
            st.write("Prediction:", result["prediction"])
        else:
            st.error("Error submitting article")

# File upload for JSON
st.header("JSON File Upload")
uploaded_file = st.file_uploader("Choose a JSON file", type="json")
if uploaded_file is not None:
    content = uploaded_file.getvalue()
    data = json.loads(content)
    if st.button("Upload JSON"):
        response = requests.post(
            f"{API_URL}/upload_json", files={"file": uploaded_file}
        )
        if response.status_code == 200:
            result = response.json()
            st.success("JSON file uploaded successfully!")
            st.write("Prediction:", result["prediction"])
        else:
            st.error("Error uploading JSON file")
