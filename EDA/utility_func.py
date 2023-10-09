import pandas as pd
import numpy as np
import surprise
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
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
    result_df_limit = result_df[result_df['count'] >= 300000].sort_values(by='star_var', ascending=False).head(5)

    return result_df_limit


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


def ScatterPlotEDA(top5_var_cities_df):
    # Create a separate figure and axes for greater customization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a color palette based on the 'count' values (scaled down by 1000)
    scaled_count = top5_var_cities_df['count'] / 1000
    colors = scaled_count

    # Create custom sizes based on the number of revisions (count)
    sizes = top5_var_cities_df['count'] / 1000  # Ajuste o fator de escala para 1000 para tornar os pontos maiores

    # Plot scatter points with custom colors and sizes
    scatter = ax.scatter(
        top5_var_cities_df['city'],
        top5_var_cities_df['star_var'],
        c=colors,
        cmap='viridis',
        s=sizes,
        alpha=0.7
    )

    # Customize labels and title
    ax.set_xlabel('City', weight='bold')
    ax.set_ylabel('Review Variance', weight='bold')
    ax.set_title('Top 5 Cities with Highest Review Variance (with more than 300k reviews)', weight='bold')

    # Rotation of x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', weight='bold')

    # Add a color bar with the label adjusted to show values without scaling
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Number of Reviews (in Thousands)', weight='bold')

    # Adjust the appearance of the chart
    ax.set_facecolor('white')
    ax.grid(False)

    # Display the graph
    plt.tight_layout()

    plt.show()



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

def Top10CategoriesByReviewsTucson(business_tucson_df, reviews_tucson_df):
    # Primeiro, mesclamos as tabelas com base no campo 'business_id'
    merged_df = pd.merge(business_tucson_df, reviews_tucson_df, on='business_id', how='inner')

    # Em seguida, criamos uma lista de todas as categorias em que cada revisão está envolvida
    all_categories = merged_df['categories'].str.split(',').explode().str.strip()

    # Calculamos o total de reviews para cada categoria
    category_review_counts = all_categories.value_counts()

    # Selecionamos as 10 principais categorias com base no total de reviews
    top_10_categories = category_review_counts.head(10)

    # Plotamos o gráfico de barras
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_10_categories.index, x=top_10_categories.values)
    plt.title('Top 10 Business Categories in Tucson by Reviews')
    plt.xlabel('Number of Reviews')
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
    business_tucson_df.drop(columns=['is_open'], inplace=True)
    correlation_matrix_business = business_tucson_df.corr()

    plt.figure(figsize = (12, 8))
    sns.heatmap(correlation_matrix_business, annot = True, cmap = 'coolwarm')
    plt.title('Correlation Matrix of Business Features in Tucson')
    plt.show()


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

