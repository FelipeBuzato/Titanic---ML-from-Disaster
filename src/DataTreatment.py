class DataTreatment:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Implement data cleaning logic here
        pass

    def transform_data(self):
        # Implement data transformation logic here
        pass

    def get_processed_data(self):
        self.clean_data()
        self.transform_data()
        return self.data
    
    #import pandas as pd
    #train = pd.read_csv('../data/train.csv')
    #print(train.head())