class KNN:
    def __init__(self, data):
        if('Survived' not in data.columns):
            raise ValueError("The 'Survived' column is required in the data for KNN.")
        
        self.y = data['Survived']
        self.x = data.drop(columns=['Survived'])

    def predict(self, k, test_data):
        # Implement prediction logic here
        pass
 
    