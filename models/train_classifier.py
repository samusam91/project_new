import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath (str): Filepath of the SQLite database.

    Returns:
    tuple: A tuple containing X (features), Y (targets), and category_names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('samtable', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text data.

    Args:
    text (str): Input text data.

    Returns:
    list: A list of tokenized words.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            if re.match('^[a-zA-Z0-9_-]*$', token):  # Check the underscore and other symbols
                clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV.

    Returns:
    GridSearchCV: A scikit-learn GridSearchCV object.
    """
    # Pipeline definition
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameters grid for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 5, 10]
    }

    # Instantiate GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model.

    Args:
    model: Trained machine learning model.
    X_test: Test features.
    Y_test: True labels for test data.
    category_names (list): List of category names.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)

    print("Classification Report:")
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Args:
    model: Trained machine learning model.
    model_filepath (str): Filepath to save the model.

    Returns:
    None
    """
    joblib.dump(model, model_filepath)
    print("Model saved to", model_filepath)


def main():
    """
    Main entry point of the script.

    This function orchestrates the execution of the data loading, model building,
    training, evaluation, and model saving processes.

    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the database and the filepath to save the model.\n'
              'Usage: python train_classifier.py <database_filepath> <model_filepath>')

if __name__ == '__main__':
    main()
