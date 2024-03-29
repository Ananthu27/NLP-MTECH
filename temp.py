from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def find_outlier_words(sentences, threshold=0.01):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    print(tfidf_df)
    outlier_words = []
    for column in tfidf_df.columns:
        # max_tfidf_score = tfidf_df[column].max()
        # min_tfidf_score = tfidf_df[column].min()

        # if max_tfidf_score > (1-threshold) or min_tfidf_score < threshold:
        #     outlier_words.append(column)

        if tfidf_df[column].sum()>(1-threshold) or tfidf_df[column].sum()<threshold:
            outlier_words.append(column)

    return outlier_words

# Example list of sentences
sentences = [
    "The cat is on the mat.",
    "A dog is sleeping.",
    "The sun is shining brightly.",
    "The cat and the dog are playing together."
]

# Specify threshold for outlier detection

# Find outlier words
outlier_words = find_outlier_words(sentences, threshold=0.01)

# Print outlier words
print("Outlier Words:")
for word in outlier_words:
    print(word)