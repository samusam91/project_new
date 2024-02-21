import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data_cat(categories_filepath):
    """
    Load category data from a CSV file.

    Args:
    categories_filepath (str): Filepath of the CSV file containing category data.

    Returns:
    DataFrame: Pandas DataFrame containing the loaded category data.
    """
    categories = pd.read_csv(categories_filepath)
    return categories


def load_data_mes(messages_filepath):
    """
    Load message data from a CSV file.

    Args:
    messages_filepath (str): Filepath of the CSV file containing message data.

    Returns:
    DataFrame: Pandas DataFrame containing the loaded message data.
    """
    messages = pd.read_csv(messages_filepath)
    return messages


def clean_data(messages, categories):
    """
    Clean the data by combining message and category dataframes.

    Args:
    messages (DataFrame): Pandas DataFrame containing message data.
    categories (DataFrame): Pandas DataFrame containing category data.

    Returns:
    DataFrame: Cleaned Pandas DataFrame containing merged message and category data.
    """
    # Merge message and category dataframes
    df = pd.merge(messages, categories, on='id')

    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Rename the columns of 'categories'
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to numeric (0 or 1)
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # Replace categories column in df with new category columns
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df



def save_data(df, database_filename):
    """
    Save the cleaned dataset to an SQLite database.

    Args:
    df (DataFrame): Pandas DataFrame containing the cleaned dataset.
    database_filename (str): Filename of the SQLite database to save the data.

    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('samtable', engine, index=False, if_exists='replace')
    print("Clean dataset saved to SQLite database.")


def main():
    """
    Main entry point of the ETL pipeline script.

    This function orchestrates the loading, cleaning, and saving of data.

    Returns:
    None
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...')
        messages = load_data_mes(messages_filepath)
        categories = load_data_cat(categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

        print('Saving data...')
        save_data(df, database_filepath)

        print('Cleaned data saved to the SQLite database!')

    else:
        print('Please provide the filepaths of the messages dataset, categories dataset, and database filepath. \n'
              'Usage: python process_data.py <messages_filepath> <categories_filepath> <database_filepath>')


if __name__ == '__main__':
    main()
