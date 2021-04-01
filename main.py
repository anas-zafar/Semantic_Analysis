import os, re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
#nltk.download('punkt')
#nltk.download('wordnet')

DIR_PATH = os.path.dirname(__file__)
PICKLES_PATH = os.path.join(DIR_PATH, "Pickles")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(debug=False)

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def welcome():
    return {"message": "Welcome, Sarcasm Detector!"}


@app.get("/is_sarcastic/{sentence}")
def is_sarcastic(sentence: str):
    _is_sarcastic = False
    _is_sarcastic = bool(pre_process_and_predict(sentence))
    return {"input_sentence": sentence, "is_sarcastic": _is_sarcastic}

@app.get("/is_sarcastic/")
def is_sarcastic_query(sentence: str):
    _is_sarcastic = False
    _is_sarcastic = bool(pre_process_and_predict(sentence))
    return {"input_sentence": sentence, "is_sarcastic": _is_sarcastic}


def pre_process_and_predict(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    # Converting the text to lower case
    text.casefold()
    # Replacing double quotes with single, within a string
    text = text.replace("\"", "\'")
    # Removing unnecessary special characters, keeping only ,  ! ? 
    text = re.sub(r"[^!?,a-zA-Z0-9\ ]+", '', text)
    # Lemmatization on verbs
    text = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in word_tokenize(text)])
    text = [text]
    print("Input", text)

    in_file = open(os.path.join(PICKLES_PATH, "vocab.pickle"), "rb")
    vocab = pickle.load(in_file)
    in_file.close()

    in_file = open(os.path.join(PICKLES_PATH, "model.pickle"), "rb")
    model = pickle.load(in_file)
    in_file.close()


    vectorizer = TfidfVectorizer(vocabulary=vocab)
    text = vectorizer.fit_transform(text)
    porbab = model.predict(text)
    print("Probab", porbab)
    return porbab[0]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
