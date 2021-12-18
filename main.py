import pandas as pd

animes = pd.read_csv('data/anime.csv')
ratings = pd.read_csv('data/rating.csv')
ratings = ratings[ratings.rating != -1]
# number of ratings
# len(ratings)
# number of users
# len(ratings['user_id'].unique())
# number of unique animes (in anime list, not ratings)
# len(animes['anime_id'].unique())
# => 11200
# avg number of anime rated per user
import statistics

ratings_per_user = ratings.groupby('user_id')['rating'].count()
statistics.mean(ratings_per_user.tolist())
# => 91.05231321839081
# distribution of ratings per user
# (we may want to exclude users without many data points)
# import matplotlib.pyplot as plt
# %matplotlib inline
# ratings_per_user.hist(bins=20, range=(0,500))

# # avg number of ratings given per anime
ratings_per_anime = ratings.groupby('anime_id')['rating'].count()
statistics.mean(ratings_per_anime.tolist())

# counts of ratings per anime as a df
ratings_per_anime_df = pd.DataFrame(ratings_per_anime)
# remove if < 1000 ratings
filtered_ratings_per_anime_df = ratings_per_anime_df[ratings_per_anime_df.rating >= 1000]
# build a list of anime_ids to keep
popular_anime = filtered_ratings_per_anime_df.index.tolist()

# counts ratings per user as a df
ratings_per_user_df = pd.DataFrame(ratings_per_user)
# remove if < 500
filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df.rating >= 500]
# build a list of user_ids to keep
prolific_users = filtered_ratings_per_user_df.index.tolist()

filtered_ratings = ratings[ratings.anime_id.isin(popular_anime)]
filtered_ratings = ratings[ratings.user_id.isin(prolific_users)]
len(filtered_ratings)

# B4
rating_matrix = filtered_ratings.pivot_table(index='user_id', columns='anime_id', values='rating')
# replace NaN values with 0
rating_matrix = rating_matrix.fillna(0)
# display the top few rows
# rating_matrix.head()

from sklearn.metrics.pairwise import cosine_similarity
import operator


def similar_users(user_id, matrix, k=3):
    # create a df of just the current user
    user = matrix[matrix.index == user_id]
    # and a df of all other users
    other_users = matrix[matrix.index != user_id]

    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(user, other_users)[0].tolist()

    # create list of indices of these users
    indices = other_users.index.tolist()

    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))

    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    # grab k users off the top
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]

    return users


current_user = 226
# # try it out
similar_user_indices = similar_users(current_user, rating_matrix)


# print(similar_user_indices)
# #=> [30773, 39021, 45603]


def recommend_item(user_index, similar_user_indices, matrix, items=5):
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    print(user_df_transposed)
    # remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating'] == 0]
    # generate a list of animes the user has not seen
    animes_unseen = user_df_transposed.index.tolist()
    print("Anime not watched yet")
    print(animes_unseen)

    # filter avg ratings of similar users for only anime the current user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(animes_unseen)]
    # order the dataframe
    similar_users_df_ordered = similar_users_df.sort_values(by=['mean'], ascending=False)
    # grab the top n anime   
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()
    # lookup these anime in the other dataframe to find names
    anime_information = animes[animes['anime_id'].isin(top_n_anime_indices)]

    return anime_information  # items


# try it out
recommend_item(226, similar_user_indices, rating_matrix)