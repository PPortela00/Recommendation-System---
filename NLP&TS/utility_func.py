from IPython.display import display
import en_core_web_sm

def YelpDatasets_Revies(reviews_df):

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
