from sklearn.model_selection import KFold
from KNN import KNN

class CrossValidation:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    
    def get_model(self, train_data):
        if(self.model == 'KNN'):
            return KNN(train_data)
        else:
            raise ValueError(f"Model {self.model} not supported for cross-validation.")
        
    
    def cross_validate(self, hyper_params):
        X = self.data

        # 10-Fold Cross-Validation
        # Split the data into 10 folds and evaluate the model on each fold
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = {hyper_param: [] for hyper_param in hyper_params}

        for train_idx, val_idx in kf.split(X):
            train = X.iloc[train_idx]
            test = X.iloc[val_idx]
            test_x = test.drop(columns=['Survived'])
            test_y = test['Survived']

            # Run the model for this fold
            model = self.get_model(train)
            model.process_test_data(test_x)

            for k in hyper_params:
                test_y_pred = model.predict(k)
                score = (test_y_pred['Survived'] == test_y).mean()
                scores[k].append(score)
            
        scores = {k: sum(v)/len(v) for k, v in scores.items()}
        return scores