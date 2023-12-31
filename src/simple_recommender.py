# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('../sample_data/movies_metadata.csv', low_memory=False)

# Print the first three rows
print(metadata.head(3))

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
print('C: ', C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print('m: ', m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print('q_movies shape: ', q_movies.shape)


# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v*R)/(v+m) + (m*C)/(m+v)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

# Print the top 15 movies
print(q_movies[['id', 'title', 'vote_count', 'vote_average', 'score']].head(20))



