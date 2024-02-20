# Disaster Response Pipeline Project

This project implements an end-to-end pipeline for analyzing and classifying disaster-related messages. It includes two main components: an ETL pipeline for data preprocessing and a machine learning pipeline for training a multi-label classification model.

# Instructions:

remember to install all the requirements with pip install

## Commands or Instructions to Run the Project

### 1. ETL Pipeline - Data Cleaning and Database Setup

To clean the data and store it in the database, run the following command in the project's root directory:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

### 2. ML Pipeline - Train Classifier and Save Model

To train the classifier and save the model, run the following command in the project's root directory:

bash

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

This command trains a machine learning model using the cleaned data from the database and saves the trained model as a pickle file (classifier.pkl).

### 3. Run Web App

After setting up the database and training the classifier, run the following command in the app's directory to launch the web app:

python run.py

Visit http://0.0.0.0:3001/ in your web browser to access the interactive web app.

Libraries Used:
- Python 3
- Pandas
- sqlalchemy
- Numpy
- sklearn
- pickle

Files Description:
- `ETL Pipeline Preparation.ipynb`: Jupyter Notebook containing the analysis of ETL pipeline
- `ML Pipeline Preparation.ipynb`: Jupyter Notebook containing the analysis of ML pipeline

Acknowledgement:
This project was created as part of a learning project and is available on GitHub for reference and collaboration.

## Link to Git : https://github.com/samusam91/project_new
