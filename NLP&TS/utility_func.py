import en_core_web_sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from IPython.display import display


def YelpDatasets_Reviews(reviews_df):

    # Display information and the first 5 rows of the 'reviews' DataFrame
    print("\nReviews DataFrame's head:")
    display(reviews_df.head())


def process_text(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    return doc


#Preprocessing data
def preprocess_sentiment_df(df):
    # change the structure of date column
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    #group by date and review_id to get the number of reviews per month and year
    df = df.groupby(['date'])['review_id'].count().reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df


def plot_df(x_pos, y_pos, x_neg, y_neg, x_df_num, y_df_num, title="", xlabel='Date', ylabel='Number of reviews', dpi=100):
    plt.figure(figsize=(15, 4), dpi=dpi)
    
    # Plot sentiment
    plt.plot(x_pos, y_pos, label='Positive', color='tab:green')
    plt.plot(x_neg, y_neg, label='Negative', color='tab:red')
    plt.plot(x_df_num, y_df_num, label='Diference (sum)', color='tab:blue')
    
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    
    # Display legend
    plt.legend()
    plt.show()

#Augmented Dickey-Fuller (ADF) test, a common statistical test for stationarity
def check_stationarity(ts):
    # ADF test
    result = adfuller(ts, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Time Series')
    plt.title('Time Series')
    plt.legend()
    plt.show()

def group_rows_by_model(data):
    grouped_data = {}
    header = data[0]
    added_metric_header = False
    
    for row in data[2:]:
        model_name = row[0].split(" (")[0] # Extract the model name without the parenthesis
        if model_name not in grouped_data:
            grouped_data[model_name] = [row]
        else:
            grouped_data[model_name].append(row)
    
    # Convert the group dictionary into a list of lists
    grouped_data_list = [header]
    for model_name, rows in grouped_data.items():
        if not added_metric_header:
            grouped_data_list.append(["", "Precision", "Recall", "F1-Score", "Support"])
            added_metric_header = True
        else:
            grouped_data_list.append(["", "", "", "", ""]) # Add a blank line as separation
        grouped_data_list.extend(rows)
    
    return grouped_data_list