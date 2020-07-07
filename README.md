# Udacity-Disaster-response

### Project Summary:
The project aims to analyze disaster data from Figure Eight(now [appen](https://appen.com))to build a model for an API that classifies disaster messages.
Project includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.

### Project Components:
1. ETL Pipeline: This Pipeline clean the raw data for later use(process_data.py)
2. ML Pipeline: This Pipeline trains and test data, the final model is exported as a pickle file(train_classifier.py)
3. Flask Web App: Visuals will be shown about analyse of the training data, also, you can enter a sos message for the Web App to classify to the 36 categories.(run.py)

### Python scripts and web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

