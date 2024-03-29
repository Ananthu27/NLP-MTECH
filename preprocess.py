# function for 1 LOWERCASING 
def to_lower(text : str) -> str :
    return text.lower()
    
import spacy

nlp = spacy.load('en_core_web_sm')
def lemmatize(string) :
    string = nlp(string)
    return ' '.join([word.lemma_ for word in string])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def remove_stopwords_predefined(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_stopwords_frequency(text,custom_stopwords):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(filtered_words)

def remove_stopwords_pos(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    filtered_words = [word for word, pos in pos_tags if pos != 'DT']  # Remove determiners
    return ' '.join(filtered_words)

import re
def remove_numbers(paragraph):
    pattern = r'\b\d+\b'
    cleaned_paragraph = re.sub(pattern, '', paragraph)
    return cleaned_paragraph

def remove_punctuations(paragraph):
    pattern = r'[^\w\s]'
    cleaned_paragraph = re.sub(pattern, '', paragraph)
    return cleaned_paragraph

def remove_single_char_words(paragraph):
    words = paragraph.split()
    filtered_words = [word for word in words if len(word) > 1]
    cleaned_paragraph = ' '.join(filtered_words)
    return cleaned_paragraph