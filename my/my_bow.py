import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor

nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def parallel_lemmatization(documents):
    with ProcessPoolExecutor() as executor:
        lemmatized_docs = list(executor.map(lemmatize_text, documents))
    return lemmatized_docs

def get_bow(documents, n_grams=(1,2,3)):
    documents = parallel_lemmatization(documents)
    vectorizer = CountVectorizer(ngram_range=n_grams, stop_words='english')
    result = vectorizer.fit_transform(documents)
    return result, vectorizer

