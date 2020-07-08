import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 

#load data from filepaths
def load_data(messages_filepath, categories_filepath):
""" 
        The function load message.csv and categories csv and return them to a merged dataframe. 
  
        Parameters: 
            the filepaths to the csv documents. 
          
        Returns: 
            the merged dataframe. 
"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id') # merge datasets
    
    return df



def clean_data(df):
""" 
        The function clean the raw dataframe and return a cleaned up dataframe. 
        It cleans the categories columns and extract to only 0 and 1 binary.
  
        Parameters: 
            the raw dataframe. 
          
        Returns: 
            cleaned dataframe. 
"""
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[slice(0,-2)])
    # rename the columns of `categories`
    categories = categories.rename(columns= category_colnames)
    for column in categories:
    # set each value to be the last character of the string
      categories[column] = categories[column].astype(str).str.slice(start=-1)
    # convert column from string to numeric
      categories[column] = categories[column].astype(int)
    # drop 204 lines where the related value is 2
      categories = categories[categories['related'] != 2 ]
    df.drop(columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join="outer")
    df.drop_duplicates( keep = False, inplace = True) 
    
    return df

# save data to database table
def save_data(df, database_filename):
""" 
        The function use engine to save the df to a database table.  
  
        Parameters: 
            the dataframe and where the table will be stocked.
          
        Returns: 
            a db table named 'DisasterResponse' 
"""
    engine = create_engine("sqlite:///{}".format(database_filepath)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()