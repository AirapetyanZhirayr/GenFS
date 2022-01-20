from nltk import word_tokenize
from nltk.corpus import stopwords
import re

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        # removes all characters beside words [a-zA-Z0-9] or whitespaces
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode()
    words = word_tokenize(text)
    words = normalize(words)
    return ' '.join(words)

