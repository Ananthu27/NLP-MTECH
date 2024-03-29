def stem(word):
    suffixes = {
        1: [("", "s", "es")],
        2: [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")],
        3: [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ],
        4: [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ],
    }

    for length in range(len(word), 0, -1):
        if length > 2:
            for old, new in suffixes[length]:
                if word.endswith(old):
                    return word[:-len(old)] + new
        else:
            return word

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