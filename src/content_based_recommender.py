# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('../sample_data/movies_metadata.csv', low_memory=False)

# print(metadata['overview'].head())

# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
# print('tfidf matrix: ', tfidf_matrix.shape)

# Array mapping from feature integer indices to feature name.
# print(tfidf.get_feature_names_out()[5000:5010])

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
consine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# print(consine_sim.shape)
# print(consine_sim[1])

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
print(indices[:10])


def get_recommendations(title, consine_sim=consine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(consine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]


print(get_recommendations('The Dark Knight Rises'))
