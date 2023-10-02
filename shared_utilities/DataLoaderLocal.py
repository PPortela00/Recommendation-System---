import pandas as pd
import sys
import os

def LoadJSON():

    # Define a nested function to read JSON data in chunks
    def read_chunks(file, cols, chunk_size=500000):
        # Use pandas read_json to read the JSON file in chunks
        # Obtenha o diretório atual do script
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Construa o caminho completo para o arquivo JSON
        json_file_path = os.path.join(current_directory, 'yelp_dataset', f'yelp_academic_dataset_{file}.json')

        # Use json_file_path como caminho para ler o arquivo JSON
        df = pd.read_json(json_file_path, chunksize=chunk_size, lines=True)

        # Create a list of DataFrames, each containing specified columns
        chunk_list = [chunk[cols] for chunk in df]

        # Concatenate the list of DataFrames into one
        return pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
    
    # Define column lists for different data types
    business_cols = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours']
    review_cols = ['review_id','user_id','business_id','stars','useful','funny','cool','text','date']
    users_cols =['user_id','name','review_count','yelping_since','useful','funny','cool','elite','friends','fans','average_stars','compliment_hot','compliment_more','compliment_profile','compliment_cute','compliment_list',
                'compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos']

    # Load data into DataFrames using the read_chunks function
    business = read_chunks('business', business_cols)
    reviews = read_chunks('review', review_cols)
    users = read_chunks('user', users_cols)

    # Return the loaded DataFrames as a tuple
    return business, reviews, users

def LoadReviews():
    # Define a nested function to read JSON data in chunks
    def read_chunks(file, cols, chunk_size=500000):
        # Obtenha o diretório atual do script
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Construa o caminho completo para o arquivo JSON
        json_file_path = os.path.join(current_directory, 'yelp_dataset', f'yelp_academic_dataset_{file}.json')

        # Use json_file_path como caminho para ler o arquivo JSON
        df = pd.read_json(json_file_path, chunksize=chunk_size, lines=True)

        # Create a list of DataFrames, each containing specified columns
        chunk_list = [chunk[cols] for chunk in df]

        # Concatenate the list of DataFrames into one
        return pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
    
    # Define column list for reviews and business
    business_cols = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours']
    review_cols = ['review_id','user_id','business_id','stars','useful','funny','cool','text','date']

    # Load review data into a DataFrame using the read_chunks function
    business = read_chunks('business', business_cols)
    reviews_df = read_chunks('review', review_cols)

    # Filter businesses in Tucson with is_open equal to 1
    business_Tucson_Open = business[(business['city'] == 'Tucson') & (business['is_open'] == 1)]

    # Now, let's select only the reviews that correspond to the filtered businesses
    reviews_df = reviews_df[reviews_df['business_id'].isin(business_Tucson_Open['business_id'])]

    # Return the loaded reviews DataFrame
    return reviews_df

