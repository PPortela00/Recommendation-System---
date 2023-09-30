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