import en_core_web_sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from IPython.display import display
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def YelpDatasets_Reviews(reviews_df):

    # Display information and the first 5 rows of the 'reviews' DataFrame
    print("\nReviews DataFrame's head:")
    display(reviews_df.head())


def PrepareDataRegression(df, n_lag_features):

    df_reg = df.copy()

    df_reg['month'] = df_reg['date'].dt.month
    df_reg['quarter'] = df_reg['date'].dt.quarter

    df_reg['yearly_sine'] = np.sin(2 * np.pi * df_reg['month'] / 12)
    df_reg['yearly_cosine'] = np.cos(2 * np.pi * df_reg['month'] / 12)

    lags = n_lag_features
    for i in range(1, lags + 1):
        df_reg[f'lag_{i}'] = df_reg['review_id'].shift(i)

    df_reg = df_reg.iloc[lags:]

    return df_reg


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


def plot_top_businesses_over_time(df, top_businesses_series):
    """
    Plot the time series of review counts for the top businesses over the years.

    Parameters:
    - df: DataFrame with columns 'business_id', 'date', 'count', and 'year'.
    - top_businesses_series: Pandas Series with the top businesses and their review counts.
    """
    # Convert the Series to a DataFrame
    top_businesses_df = top_businesses_series.reset_index()

    # Filter the DataFrame to include only the top businesses
    top_businesses_df = df[df['business_id'].isin(top_businesses_df['business_id'])]

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set_palette("viridis")  # You can choose a different color palette if needed

    for business_id in top_businesses_df['business_id'].unique():
        business_data = top_businesses_df[top_businesses_df['business_id'] == business_id]
        sns.lineplot(x='date', y='count', data=business_data, label=f'Business {business_id}')

    plt.title('Time Series of Review Counts for Top Businesses Over the Years', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Review Counts', fontsize=14)
    plt.legend()
    
    # Set y-axis limits to start from 0
    plt.ylim(bottom=0)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    plt.show()


def SeasonalPlot(df, years_to_show=[]):
    df_1 = df.copy()

    if len(years_to_show) != 0:
        # Filter the DataFrame to include only the desired years
        df_1 = df_1[df_1['year'].isin(years_to_show)]

    # Pivot the filtered DataFrame to have years as columns and months as the index
    pivot_df = df_1.pivot(index='month', columns='year', values='count')

    # Create a seasonal plot
    plt.figure(figsize=(12, 6))
    plt.title('Seasonal Plot')
    plt.xlabel('Month')
    plt.ylabel('Value')

    # Plot each year's data as a separate line
    for year in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[year], label=str(year))

    # Add legend
    plt.legend(loc='best')

    # Customize x-axis ticks (e.g., month names)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(pivot_df.index, month_names)

    # Show the plot
    plt.grid(True)
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


def ExponentialSmoothingModeling(model_type, sentiment_data, alpha=0.9, beta=0.9, gamma=0.9, decomposition_type='additive', damped=False):

    if model_type == 'simple':
        model = SimpleExpSmoothing(endog = sentiment_data['review_id'])
        results = model.fit(smoothing_level = alpha)

    elif model_type == 'holt':
        model = Holt(endog = sentiment_data['review_id'], 
                     damped_trend = damped)

        results = model.fit(smoothing_level = alpha,
                            smoothing_trend = beta)
        
    else:
        model = ExponentialSmoothing(endog = sentiment_data['review_id'], 
                                     trend = decomposition_type, 
                                     seasonal = decomposition_type, 
                                     seasonal_periods = 12, 
                                     damped_trend = damped)

        results = model.fit(smoothing_level = alpha, 
                            smoothing_trend = beta, 
                            smoothing_seasonal = gamma)
        
    return model, results


def EvaluateBaseline(baseline_value, y_true):

    y_hat = np.array([baseline_value] * len(y_true))
    rmse = mean_squared_error(y_true, y_hat, squared = False)
    mae = mean_absolute_error(y_true, y_hat)
    mape = mean_absolute_percentage_error(y_true, y_hat)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}')

    return y_hat


def EvaluateETS(model, y_true):

    y_hat = np.array(model.forecast(13))

    rmse = mean_squared_error(y_true, y_hat, squared = False)
    mae = mean_absolute_error(y_true, y_hat)
    mape = mean_absolute_percentage_error(y_true, y_hat)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}')

    return y_hat


def EvaluateRegression(model, X_test, y_true):

    y_hat = model.predict(X_test)

    rmse = mean_squared_error(y_true, y_hat, squared = False)
    mae = mean_absolute_error(y_true, y_hat)
    mape = mean_absolute_percentage_error(y_true, y_hat)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}')

    return y_hat


def PlotPredictions(date, y_true, y_hat, title, xlabel='Dates', ylabel='Number of reviews'):
    plt.figure(figsize=(15, 4), dpi=300)
    
    # Plot sentiment
    plt.plot(date, y_true, label='True', color='tab:green')
    plt.plot(date, y_hat, label='Predicted', color='tab:red')
        
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        
    # Display legend
    plt.legend()
    plt.show()
