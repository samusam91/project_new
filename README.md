# Disaster Response Pipeline Project

This project implements an end-to-end pipeline for analyzing and classifying disaster-related messages. 
It includes two main components: an ETL pipeline for data preprocessing and a machine learning pipeline for training a multi-label classification model.

# Importance and Impact:
In times of disaster, effective communication is critical for coordinating response efforts and providing assistance to those in need. 
However, the overwhelming volume of messages generated during such events can make it challenging for responders to prioritize and address urgent requests.

The Disaster Response Pipeline Project aims to alleviate this challenge by providing a robust system for automatically categorizing and prioritizing incoming messages based on their relevance and urgency.
By quickly identifying messages pertaining to specific needs such as medical assistance, food supplies, or shelter, responders can streamline their efforts and allocate resources more efficiently.

# Practical Benefits:
Faster Response Times: With the automated classification system, responders can swiftly identify critical messages, leading to faster response times and potentially saving lives.
Resource Allocation: By accurately categorizing messages, the system helps organizations allocate resources where they are most needed, optimizing the use of limited resources during a crisis.
Improved Coordination: The centralized platform fosters better coordination among different response teams and organizations, ensuring a more cohesive and effective disaster response effort.
Enhanced Situational Awareness: By analyzing trends and patterns in incoming messages, the system provides valuable insights into the evolving needs and priorities of affected communities, enabling responders to adapt their strategies accordingly.

# Project structure:
- Project_new
    - app
        - templates
            - go.html
            - master.html
        - data
            - disaster_categories.csv
            - disaster_messages.csv
            - process_data.py
        - models
            - train_classifier.py
    - run.py
    - disaster_response.db

# Instructions:
Remember to install all the requirements with pip install.

## Commands to Run the Project

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

## Link to Git: 
Disaster Response Pipeline Project on GitHub: https://github.com/samusam91/project_new
