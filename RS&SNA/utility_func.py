import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import surprise

from sklearn.manifold import TSNE
from IPython.display import display
from lightfm import LightFM
from lightfm.data import Dataset
from itertools import combinations


def YelpDatasets(business_df, reviews_df, users_df):
    # Display information and the first 5 rows of the 'business' DataFrame
    print("Business DataFrame's head:")
    display(business_df.head())

    # Display information and the first 5 rows of the 'reviews' DataFrame
    print("\nReviews DataFrame's head:")
    display(reviews_df.head())

    # Display information and the first 5 rows of the 'users' DataFrame
    print("\nUsers DataFrame's head:")
    display(users_df.head())


def NumReviewsBusiness(df_reviews):

    business_counts = df_reviews['business_id'].value_counts()

    plt.figure(figsize=(8, 6))
    business_counts.hist(bins=500)
    plt.xlabel('Number of reviews')
    plt.ylabel('Frequency')
    plt.title('Business Frequency Histogram')

    # Set the x-axis limits to zoom in
    plt.xlim(left=0, right=100)  # Adjust the range as needed

    # Show the histogram
    plt.show()

    return business_counts


def ConvertStringKeyToIntegerKey(df, col):
    # Get unique strings from the column
    unique_strings = df[col].unique()

    # Create a mapping dictionary to assign unique integer keys
    string_to_int_mapping = {string: index for index, string in enumerate(unique_strings)}

    # Add a new column with integer keys
    df[col] = df[col].map(string_to_int_mapping)

    return df


def PrepareDataFrameRS(business_df, reviews_df, users_df, city, business_to_filter):
    # Aggregation of data, with relevant columns, from the business and reviews datasets
    business_cols = ['business_id', 'city', 'is_open']
    reviews_cols = ['review_id', 'user_id', 'business_id', 'stars']
    users_cols = ['user_id']

    # Merge the relevant dataframes
    df = pd.merge(reviews_df[reviews_cols], business_df[(business_df['is_open'] == 1) & (business_df['city'] == city)][business_cols], 
                  left_on='business_id', right_on='business_id', how='inner')
    df = pd.merge(df, users_df[users_cols], left_on='user_id', right_on='user_id', how='left')
    df = df[['user_id', 'business_id', 'stars']]

    df = df[~df['business_id'].isin(business_to_filter)]

    df = ConvertStringKeyToIntegerKey(df, 'user_id')
    df = ConvertStringKeyToIntegerKey(df, 'business_id')

    # Remove businesses with a number of reviews equal to or less than 1
    business_review_counts = df['business_id'].value_counts()
    businesses_to_keep = business_review_counts[business_review_counts > 1].index
    df = df[df['business_id'].isin(businesses_to_keep)]

    # Remove duplicate entries
    df = df.drop_duplicates(subset=['user_id', 'business_id'], keep="first", inplace=False)
    
    return df


def PrepareDataSurprise(df, sample_size=326439):
    # Create a Surprise Reader specifying the rating scale
    reader = surprise.Reader(rating_scale=(df.stars.min(), df.stars.max()))
    # Load the pandas DataFrame into a Surprise Dataset
    data = surprise.Dataset.load_from_df(df[['user_id', 'business_id', 'stars']].sample(sample_size), reader)

    trainset, testset = surprise.model_selection.train_test_split(data, test_size=0.01)

    return trainset, testset


def PrepareDataLightFM(full_dataset, trainset, testset):
    # Extract all rows from the Trainset
    rows = [(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid), rating) for (uid, iid, rating) in trainset.all_ratings()]

    # Create a pandas DataFrame from the extracted rows
    train = pd.DataFrame(rows, columns=['user_id', 'business_id', 'stars']) 
    test = pd.DataFrame(testset, columns=['user_id', 'business_id', 'stars'])

    train_dataset = Dataset()
    train_dataset.fit(full_dataset.user_id,full_dataset.business_id)

    (train_interactions, train_weights) = train_dataset.build_interactions([(x['user_id'],
                                                                             x['business_id'],
                                                                             x['stars']) for index,x in train.iterrows()])
    
    train = (train_interactions, train_weights)
    
    return train, test


def ComputeCombinations(df_sna, n_combinations):
    # Calculate weights and add edges efficiently
    unique_users = df_sna['user_id'].unique()
    user_combinations = np.array(list(combinations(unique_users, 2)))
    np.random.shuffle(user_combinations)

    user_combinations = user_combinations[:n_combinations]

    return user_combinations


def PrepareDataSNA(combinations, np_sna):
    user1_ids, user2_ids = combinations[:, 0], combinations[:, 1]

    # Create a dictionary to map unique user IDs to their positions in sna_np
    user_id_to_index = {}
    for index, user_id in enumerate(np_sna[:, 0]):
        if user_id not in user_id_to_index:
            user_id_to_index[user_id] = index

    # Map user IDs to their positions
    user1_indexes = np.array([user_id_to_index[user_id] for user_id in user1_ids])
    users1 = np_sna[user1_indexes]

    return users1, user2_ids


# Define evaluation function
def evaluate_algorithm(algo, trainset, testset):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Compute and return RMSE
    rmse = surprise.accuracy.rmse(predictions)
    return rmse, predictions


def RandomRecommender(data_train, data_test):
    # Random Recommender
    random_algo = surprise.NormalPredictor()
    random_rmse, predictions = evaluate_algorithm(random_algo, data_train, data_test)

    return random_algo, random_rmse, predictions


def UserBasedCollaborativeFiltering(data_train, data_test):
    ubcf_algo = surprise.KNNBasic(sim_options={'user_based': True})
    ubcf_rmse, predictions = evaluate_algorithm(ubcf_algo, data_train, data_test)

    return ubcf_algo, ubcf_rmse, predictions


def ItemBasedCollaborativeFiltering(data_train, data_test):
    ibcf_algo = surprise.KNNBasic(sim_options={'user_based': False})
    ibcf_rmse, predictions = evaluate_algorithm(ibcf_algo, data_train, data_test)

    return ibcf_algo, ibcf_rmse, predictions


def SingularValueDecomposition(data_train, data_test, n_factors=100, n_epochs=20, lr=0.005, reg_all=0.02):
    svd_algo = surprise.SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg_all)
    svd_rmse, predictions = evaluate_algorithm(svd_algo, data_train, data_test)

    return svd_algo, svd_rmse, predictions


def SingularValueDecompositionPP(data_train, data_test, n_factors=100, n_epochs=20, lr=0.005, reg_all=0.02, cache_ratings = False):
    svdpp_algo = surprise.SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg_all, cache_ratings=cache_ratings)
    svdpp_rmse, predictions = evaluate_algorithm(svdpp_algo, data_train, data_test)

    return svdpp_algo, svdpp_rmse, predictions


def RMSE(y_true, y_pred):
    # Calculate the squared differences between the two arrays
    squared_diff = (y_true - y_pred) ** 2

    # Calculate the mean of squared differences
    mean_squared_diff = squared_diff.mean()

    # Calculate the RMSE by taking the square root of the mean squared difference
    rmse = np.sqrt(mean_squared_diff)

    return rmse


def MatrixFactorizationLightFM(train, test, n_components=10, learning_schedule='adagrad', learning_rate=0.05, loss_func='warp', epochs=20):
    model = LightFM(no_components=n_components, 
                    learning_schedule=learning_schedule, 
                    learning_rate=learning_rate,
                    loss=loss_func)
    
    model.fit(interactions=train[0], user_features=None, item_features=None, sample_weight=train[1], epochs=epochs)

    test_user_ids = np.array(test['user_id'])
    test_item_ids = np.array(test['business_id'])

    y_hat = model.predict(test_user_ids, test_item_ids)
    y_true = np.array(test['stars'])

    rmse = RMSE(y_true, y_hat)
    print(f"RMSE: {rmse:.4f}")

    return model, rmse, y_hat


def PredictionsRS(trainset, predictions, n):
    def get_Iu(uid):
        """ return the number of items rated by given user
        args: 
        uid: the id of the user
        returns: 
        the number of items rated by the user
        """
        try:
            return len(trainset.ur[trainset.to_inner_uid(uid)])
        except ValueError: # user was not part of the trainset
            return 0
        
    def get_Ui(iid):
        """ return number of users that have rated given item
        args:
        iid: the raw id of the item
        returns:
        the number of users that have rated the item.
        """
        try: 
            return len(trainset.ir[trainset.to_inner_iid(iid)])
        except ValueError:
            return 0
        
    df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    df['user_n_rated'] = df.uid.apply(get_Iu)
    df['item_n_rated'] = df.iid.apply(get_Ui)
    df['err'] = abs(df.est - df.r_ui)
    best_predictions = df.sort_values(by = 'err')[:n]
    worst_predictions = df.sort_values(by = 'err')[-n:]

    return df, best_predictions, worst_predictions


# Recommend top N items for a user using a recommender model
def recommend_top_n(algo, trainset, raw_user_id, n=10):
    user_ratings = trainset.ur[trainset.to_inner_uid(raw_user_id)]
    items = [item_id for (item_id, _) in user_ratings]
    
    item_scores = {}
    for item_id in trainset.all_items():
        if item_id not in items:
            raw_item_id = trainset.to_raw_iid(item_id)
            prediction = algo.predict(raw_user_id, raw_item_id, verbose=False)
            item_scores[raw_item_id] = prediction.est
    
    top_items = sorted(item_scores, key = item_scores.get, reverse = True)[:n]

    return top_items, item_scores


def perform_tsne(svd_matrix, n_componets, n_iter):
    tsne = TSNE(n_components=n_componets, n_iter=n_iter, verbose=3, random_state=1)
    res_embedding = tsne.fit_transform(svd_matrix)
    projection = pd.DataFrame(columns=['x', 'y'], data=res_embedding)
    return projection


"""This function takes information about a user (user1) and calculates their 'elite' status. 
It assumes that the user's elite status is represented as a comma-separated string in the 'elite' column of the DataFrame. 
It counts the number of elements in the 'elite' string and returns the minimum of that count and 5."""
def EliteUsers(user1):
    
    value = len(user1[4].split(','))

    return min(value, 5)

"""This function checks if a given user ID (user) is present in a list of friends (user_list).
 If the user is in the list, it returns 1 (indicating a friendship connection); otherwise, it returns 0."""
def Friends(user, user_list):
    if user in user_list:
        return 1
    else:
        return 0
    
"""This function counts the number of businesses that have been reviewed by both user1 and user2 in the given DataFrame df. 
It filters the DataFrame to include only reviews by these two users, then counts the unique businesses they have reviewed together."""""
def BusinessesReviewedCommom(user1, user2, sna_numpy):

    user_mask = np.logical_or(sna_numpy[:, 0] == user1, sna_numpy[:, 0] == user2)
    filtered_array = sna_numpy[user_mask]

    # Extract the 'business_id' column from the filtered array
    business_ids = filtered_array[:, 1]

    # Get the unique values and their counts
    unique_values, counts = np.unique(business_ids, return_counts=True)

    value = np.sum(counts == 2)

    return value


"""This function checks if user1 and user2 have given the same review rating for any businesses in the DataFrame df. 
It does this by comparing the star ratings given by these users for the businesses they have reviewed."""
def SameReviewRating(user1, user2, df):

    rating1 = df[df['user_id'] == user1].stars
    rating2 = df[df['user_id'] == user2].stars

    return rating1 == rating2

"""This is the main function that calculates the weight of the connection between user1_id and user2_id. It combines the results from the other functions:
It calculates the "elite" status of user1_id using the EliteUsers function.
It adds a contribution based on the number of fans of user1_id divided by 100.
It checks if user2_id is in the list of friends of user1_id using the Friends function.
It counts the number of businesses reviewed in common by both users using the BusinessesReviewedCommom function."""
def Weight_v1(user1_id, user2_id, sna_numpy):

    user_1 = sna_numpy[sna_numpy[:, 0] == user1_id][0]
    friends = user_1[2].split(', ')

    return EliteUsers(user_1) + (user_1[3] / 100) + Friends(user2_id, friends) + BusinessesReviewedCommom(user1_id, user2_id, sna_numpy)



"""An alternative to calculating a user's 'elite' status could be to consider the duration of elite status.
Instead of counting the number of comma-separated elements in the 'elite' column, you could calculate the number of years the user has held elite status."""
def EliteUsers_v1(user1):
    elite_status = user1['elite'].apply(lambda x: x.split(','))
    elite_years = [int(year.strip()) for x in elite_status for year in x if year.strip().isdigit()]
    return len(elite_years)

"""An alternative way of verifying the existence of direct friendships between users could be to calculate the total number of friends in common between two users. 
This would take into account the strength of the connection between them based on how many friends they share in common."""
# Not working
def Friends_v1(user1, user2):
    friends_user1 = set(user1['friends'].split(', '))
    friends_user2 = set(user2['friends'].split(', '))
    common_friends = friends_user1.intersection(friends_user2)
    return len(common_friends)

def BusinessesReviewedCommom_v1(user1, user2, df):
    df_user1 = df[(df['user_id'] == user1) & df['business_id'].isin(df[df['user_id'] == user2]['business_id'])]
    df_user2 = df[(df['user_id'] == user2) & df['business_id'].isin(df[df['user_id'] == user1]['business_id'])]

    if df_user1.empty or df_user2.empty:
        return 0.0  # Não há avaliações em comum

    avg_rating_diff = (df_user1['stars'] - df_user2['stars']).mean()
    return avg_rating_diff

def Weight_v2(user1_id, user2_id, users_df, df):
    user_1 = users_df[users_df['user_id'] == user1_id]
    friends = user_1.reset_index().drop(columns=['index'])['friends'][0].split(', ')

    return EliteUsers_v1(user_1) + (user_1['fans'] / 100) + Friends_v1(user2_id, friends) + BusinessesReviewedCommom_v1(user1_id, user2_id, df)


def Weight(users_1, users_2):
    def EliteUsers(users_1):
    
        values = np.array([len(elite.split(',')) for elite in users_1[:, 4]])

        return np.minimum(values, 5)
    

    def FansCount(users_1):

        return users_1[:, 3] / 100


    def Friends(users_1, users_2_ids):
        def Friend(users_2, user_list):
            return np.max(np.where(np.isin(user_list, users_2), 1, 0))

        # Compute friends for all users
        friends_array = np.array(list(friends.split(', ') for friends in users_1[:, 2]), dtype=object)
        all_friends = np.array(list(Friend(user2, friends1) for user2, friends1 in zip(users_2_ids, friends_array)))

        return all_friends
    

    return EliteUsers(users_1) + FansCount(users_1) + Friends(users_1, users_2)
