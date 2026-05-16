import pandas as pd

class DataTreatment:
    def __init__(self, data):
        self.data = data.copy()
        self.original_data = data.copy()
        self.ticket_survival_rate = None

    def clean_data(self):
        ## Implement data cleaning logic here
        
        # 1. We'll drop the PassengerId and Name columns as they are not useful for our analysis
        #    We'll also drop the Cabin column as it has too many missing values
        self.data = self.data.drop(columns=['PassengerId', 'Name', 'Cabin'], errors='ignore')

        # 2. We'll replace the Nans with the most frequent value in the column
        self.data = self.data.fillna(self.data.mode().iloc[0])


    def transform_data(self):
        ## Implement data transformation logic here

        # 1. Convert Sex and Embarked columns to numerical variables using using one-hot encoding (no ordering between labels)
        transf_data = pd.get_dummies(self.data, columns=['Sex', 'Embarked'], drop_first=True)

        # 2. Transform the Ticket column
        #    Ticket Frequency can tell us how many people made the same booking as the passenger
        transf_data['Ticket_freq'] = transf_data['Ticket'].map(transf_data['Ticket'].value_counts())
        
        # We can also get the prefix of the ticket, which can give us info about the type of ticket
        transf_data['Ticket_prefix'] = transf_data['Ticket'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'NO_PREFIX')

        # Create ticket score, which is the probability of survival for each ticket prefix
        if('Survived' in transf_data.columns):
            self.ticket_survival_rate = transf_data.groupby('Ticket_prefix')['Survived'].mean()
        if(self.ticket_survival_rate is None):
            raise ValueError("Set the ticket survival rate in the test set to the one from the training set.")
        
        transf_data['Ticket_score'] = transf_data['Ticket_prefix'].map(self.ticket_survival_rate)
        transf_data['Ticket_score'] = transf_data['Ticket_score'].fillna(transf_data['Ticket_score'].mode().iloc[0])
        transf_data = transf_data.drop(columns=['Ticket', 'Ticket_prefix'], errors='ignore')

        # 3. Standardize the data
        survived_col = transf_data['Survived'] if 'Survived' in transf_data.columns else None
        transf_data = transf_data.drop(columns=['Survived'], errors='ignore')
        transf_data = (transf_data - transf_data.mean()) / transf_data.std()
        
        if(survived_col is not None):
            transf_data['Survived'] = survived_col
        
        self.data = transf_data


    def get_processed_data(self):
        self.clean_data()
        self.transform_data()
        return self.data
    
    
"""
import pandas as pd
train = pd.read_csv('../data/train.csv')
#print(train.head())
data_treatment = DataTreatment(train)
processed_data = data_treatment.get_processed_data()
#print(processed_data.head())
#print(data_treatment.ticket_survival_rate)
"""
    