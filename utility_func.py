import pandas as pd
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
