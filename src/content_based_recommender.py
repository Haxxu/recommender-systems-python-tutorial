# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('../sample_data/movies_metadata.csv', low_memory=False)

print(metadata['overview'].head())

# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
print('tfidf matrix: ', tfidf_matrix.shape)


# Array mapping from feature integer indices to feature name.
print(tfidf.get_feature_names_out()[5000:5010])



