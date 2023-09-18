import pandas as pd
import surprise
import matplotlib.pyplot as plt


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


def PrepareDataSurprise(df, sample_size=100000):
    # Create a Surprise Reader specifying the rating scale
    reader = surprise.Reader(rating_scale=(df.stars.min(), df.stars.max()))
    # Load the pandas DataFrame into a Surprise Dataset
    data = surprise.Dataset.load_from_df(df[['user_id', 'business_id', 'stars']].sample(sample_size), reader)

    trainset, testset = surprise.model_selection.train_test_split(data, test_size=0.1)

    return trainset, testset


# Define evaluation function
def evaluate_algorithm(algo, trainset, testset):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Compute and return RMSE
    rmse = surprise.accuracy.rmse(predictions)
    return rmse


def UserBasedCollaborativeFiltering(data_train, data_test):
    ubcf_algo = surprise.KNNBasic(sim_options={'user_based': True})
    ubcf_rmse = evaluate_algorithm(ubcf_algo, data_train, data_test)

    return ubcf_algo, ubcf_rmse