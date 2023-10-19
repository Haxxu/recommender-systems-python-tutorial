import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

metadata = pd.read_csv('../store_data/keywords_based.csv')


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


def get_recommendations(title, consine_sim=cosine_sim2):
    idx = indices[title]
    sim_scores = list(enumerate(consine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]



print(get_recommendations('The Dark Knight Rises', cosine_sim2))