from DataTreatment import DataTreatment
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, data):
        if('Survived' not in data.columns):
            raise ValueError("The 'Survived' column is required in the data for KNN.")
        
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


    def predict(self, k):
        # Prediction
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.x_train, self.y_train)
        predictions = knn.predict(self.x_test)

        # Submission format
        passenger_ids = self.test_data_processor.original_data['PassengerId']
        predictions = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})

        return predictions
    


import pandas as pd
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
#print(train.head())
#data_treatment = DataTreatment(train)
#processed_data = data_treatment.get_processed_data()
#print(processed_data.head())
knn = KNN(train)
knn.process_test_data(test)
predictions =knn.predict(k=5)
print(predictions)