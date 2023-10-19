# Import Pandas
import pandas as pd
import numpy as np

# Load Movies Metadata
metadata = pd.read_csv('../sample_data/movies_metadata.csv', low_memory=False)

# Load keywords and credits
credits = pd.read_csv('../sample_data/credits.csv')
keywords = pd.read_csv('../sample_data/keywords.csv')

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Convert IDs to int. Required for merging
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# print(metadata.head(2))

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names


metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Print the new features of the first 3 films
# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)

print(metadata[['soup']].head(2))

metadata.to_csv('../store_data/keywords_based.csv')
