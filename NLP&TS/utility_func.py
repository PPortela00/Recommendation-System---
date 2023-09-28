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


