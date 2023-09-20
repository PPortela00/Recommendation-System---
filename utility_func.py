import pandas as pd
import surprise
import matplotlib.pyplot as plt
import seaborn as sns

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


def OpenClosedTucson(business_df):
    # 2. Count of Open and Closed Businesses in Tucson:
    df_business_tucson = business_df[business_df['city'] == 'Tucson']

    # Count the number of open and closed businesses in Tucson
    business_open_closed = df_business_tucson['is_open'].value_counts()

    # Create a bar plot
    plt.figure(figsize = (8, 6))
    sns.barplot(x = business_open_closed.index, y = business_open_closed.values, palette = 'Set2')
    plt.xticks([0, 1], ['Closed', 'Open'])
    plt.title('Count of Open and Closed Businesses in Tucson')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.show()

    return df_business_tucson


def Top10CommomCategoriesTucson(business_tucson_df):
    plt.figure(figsize = (12, 8))
    top_categories = business_tucson_df['categories'].str.split(',').explode().str.strip().value_counts()[:10]
    sns.barplot(y = top_categories.index, x = top_categories.values)
    plt.title('Top 10 Business Categories in Tucson')
    plt.xlabel('Number of Businesses')
    plt.ylabel('Category')
    plt.show()


def UserRegistrationReviewsTucson(users_df, reviews_tucson_df):
    # 4. Analysis of User Registration Dates related to reviews in Tucson:
    df_users_tucson = users_df[users_df['user_id'].isin(reviews_tucson_df['user_id'])]
    df_users_tucson['yelping_since'] = pd.to_datetime(df_users_tucson['yelping_since'])
    df_users_tucson['yelping_since_year'] = df_users_tucson['yelping_since'].dt.year

    plt.figure(figsize = (10, 6))
    sns.histplot(data = df_users_tucson, x = 'yelping_since_year', bins = 20, kde = True)
    plt.title('Distribution of User Registration Dates in Tucson (Related to Reviews)')
    plt.xlabel('Registration Year')
    plt.ylabel('Count')
    plt.show()


def CorrelationMatrix(business_tucson_df):
    # 5. Correlation between Features (only for numeric variables) in the "business" table:
    correlation_matrix_business = business_tucson_df.corr()

    plt.figure(figsize = (12, 8))
    sns.heatmap(correlation_matrix_business, annot = True, cmap = 'coolwarm')
    plt.title('Correlation Matrix of Business Features in Tucson')
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

    #df = ConvertStringKeyToIntegerKey(df, 'user_id')
    #df = ConvertStringKeyToIntegerKey(df, 'business_id')

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


def SingularValueDecompositionPP(data_train, data_test, n_factors=20, n_epochs=20, lr=0.007, reg_all=0.02, cache_ratings=False):
    svdpp_algo = surprise.SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg_all, cache_ratings=cache_ratings)
    svdpp_rmse, predictions = evaluate_algorithm(svdpp_algo, data_train, data_test)

    return svdpp_algo, svdpp_rmse, predictions


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


# Recommend top N items for a user using a recommender model
def recommend_top_n(algo, trainset, user_id, n=10):
    user_ratings = trainset.ur[trainset.to_inner_uid(user_id)]
    items = [item_id for (item_id, _) in user_ratings]
    
    item_scores = {}
    for item_id in trainset.all_items():
        if item_id not in items:
            prediction = algo.predict(user_id, trainset.to_raw_iid(item_id), verbose=False)
            item_scores[item_id] = prediction.est
    
    top_items = sorted(item_scores, key = item_scores.get, reverse=True)[:n]

    #from raw_id to actual_id
    return [trainset.to_raw_iid(i) for i in top_items], item_scores
