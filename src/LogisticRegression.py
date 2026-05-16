from sklearn.linear_model import LogisticRegression as SktLogisticRegression
import pandas as pd
from DataTreatment import DataTreatment

class LogisticRegression:
    def __init__(self, data):
        if('Survived' not in data.columns):
            raise ValueError("The 'Survived' column is required in the data for Logistic Regression.")
        
        self.train_data_processor = DataTreatment(data)
        self.test_data_processor = None

        # process the training data
        self.train = self.train_data_processor.get_processed_data()
        self.x_train = self.train.values[:, :-1]
        self.y_train = self.train.values                                                                                                                                                                                                                                                                                                                         [:, -1]


    def process_test_data(self, test):
        self.test_data_processor = DataTreatment(test)
        self.test_data_processor.ticket_survival_rate = self.train_data_processor.ticket_survival_rate
        self.test = self.test_data_processor.get_processed_data()
        self.x_test = self.test.values

    
    def predict(self, params=None):
        if(params is None):
            l1_ratio = 0 # L2 regularization
            c = 1.0
        else:
            l1_ratio = params[0]
            c = params[1]

        log_reg = SktLogisticRegression(solver='saga', max_iter=1000, l1_ratio=l1_ratio, C=c)
        log_reg.fit(self.x_train, self.y_train)
        predictions = log_reg.predict(self.x_test)
    
         # Submission format
        passenger_ids = self.test_data_processor.original_data['PassengerId']
        predictions = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})

        return predictions