import pandas as pd
import surprise
import matplotlib.pyplot as plt

from IPython.display import display


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


def Top5VarianceCities(merged_df):
    # Compute the variance of ratings for each city
    star_variance_by_city = merged_df.groupby('city')['stars'].var().reset_index()
    star_variance_by_city.columns = ['city', 'star_var']

    # Compute the count of each city in the original DataFrame
    city_count = merged_df['city'].value_counts().reset_index()
    city_count.columns = ['city', 'count']

    # Merge the variance and count DataFrames on the 'City' column
    result_df = pd.merge(star_variance_by_city, city_count, on='city')

    # Sort the results by variance in descending order
    result_df = result_df.sort_values(by='star_var', ascending=False)

    # Choosing the cities with higher variance and with more than 300k reviews
    result_df_limit5 = result_df[result_df['count'] >= 300000].sort_values(by='star_var', ascending=False).head(5)

    return result_df_limit5


# This function describes the review's stars of a specified city
def DescribeCity(merged_df, city):
    city_desc = merged_df[merged_df['city'] == city][['stars']].describe()
    city_desc[city] = city_desc['stars']
    city_desc.drop(columns=['stars'], inplace=True)

    return city_desc


# This function describes the review's stars of a set of cities specified in a list
def DescribeCities(merged_df, cities):
    city_0 = DescribeCity(merged_df, cities[0])
    city_1 = DescribeCity(merged_df, cities[1])
    city_2 = DescribeCity(merged_df, cities[2])
    city_3 = DescribeCity(merged_df, cities[3])
    city_4 = DescribeCity(merged_df, cities[4])

    city_describe = city_0
    city_describe[cities[1]] = city_1
    city_describe[cities[2]] = city_2
    city_describe[cities[3]] = city_3
    city_describe[cities[4]] = city_4

    return city_describe


# This function plots an histogram of the review's ratings
def HistogramEDA2_2(merged_df, city):

    merged_df[merged_df['city'] == city][['stars']].hist()

    plt.title(city)
    plt.xlabel('Stars')
    plt.ylabel('Number of reviews')
    plt.show()


def ConvertStringKeyToIntegerKey(df, col):
    # Get unique strings from the column
    unique_strings = df[col].unique()

    # Create a mapping dictionary to assign unique integer keys
    string_to_int_mapping = {string: index for index, string in enumerate(unique_strings)}

    # Add a new column with integer keys
    df[col] = df[col].map(string_to_int_mapping)

    return df


def PrepareDataFrameRS(business_df, reviews_df, users_df, city):
    # Aggregation of data, with relevant columns, from the business and reviews datasets
    business_cols = ['business_id', 'city', 'is_open']
    reviews_cols = ['review_id', 'user_id', 'business_id', 'stars']
    users_cols = ['user_id']

    df = pd.merge(reviews_df[reviews_cols], business_df[(business_df['is_open'] == 1) & (business_df['city'] == city)][business_cols], 
                  left_on='business_id', right_on='business_id', how='inner')
    df = pd.merge(df, users_df[users_cols], left_on='user_id', right_on='user_id', how='left')
    df = df[['user_id', 'business_id', 'stars']]

    df = ConvertStringKeyToIntegerKey(df, 'user_id')
    df = ConvertStringKeyToIntegerKey(df, 'business_id')

    df = df.drop_duplicates(subset=['user_id', 'business_id'], keep="first", inplace=False)
    
    return df


def PrepareDataSurprise(df, sample_size=326439):
    # Create a Surprise Reader specifying the rating scale
    reader = surprise.Reader(rating_scale=(df.stars.min(), df.stars.max()))
    # Load the pandas DataFrame into a Surprise Dataset
    data = surprise.Dataset.load_from_df(df[['user_id', 'business_id', 'stars']].sample(sample_size), reader)

    trainset, testset = surprise.model_selection.train_test_split(data, test_size=0.01)

    return trainset, testset


# Define evaluation function
def evaluate_algorithm(algo, trainset, testset):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Compute and return RMSE
    rmse = surprise.accuracy.rmse(predictions)
    return rmse, predictions


def UserBasedCollaborativeFiltering(data_train, data_test):
    ubcf_algo = surprise.KNNBasic(sim_options={'user_based': True})
    ubcf_rmse, predictions = evaluate_algorithm(ubcf_algo, data_train, data_test)

    return ubcf_algo, ubcf_rmse, predictions


def ItemBasedCollaborativeFiltering(data_train, data_test):
    ibcf_algo = surprise.KNNBasic(sim_options={'user_based': False})
    ibcf_rmse, predictions = evaluate_algorithm(ibcf_algo, data_train, data_test)

    return ibcf_algo, ibcf_rmse, predictions


def SingularValueDecomposition(data_train, data_test, n_factors=100, n_epochs=20, lr=0.005):
    svd_algo = surprise.SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr)
    svd_rmse, predictions = evaluate_algorithm(svd_algo, data_train, data_test)

    return svd_algo, svd_rmse, predictions


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
    best_predictions = df.sort_values(by='err')[:n]
    worst_predictions = df.sort_values(by='err')[-n:]

    return df, best_predictions, worst_predictions
