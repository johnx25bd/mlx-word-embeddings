# Predicting Hacker News upvotes

This project is a simple API that predicts the number of upvotes a Hacker News article will receive. 

A variation of word2vec is built from scratch to create word embeddings from text8, a wiki dataset, these embeddings are fine-tuned on the Hacker News titles. 

These embeddings, along with a some other features, are used to train a predictor model. The user can interact with the model through a streamlit app by providing title, author, url, and date.

## Setup

Create virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies (python 3.12):

```bash
pip install -r requirements.txt
``` 

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Start the streamlit app:

```bash
streamlit run app/streamlit_app.py
```



