import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            return
        
        # Load training data
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']].values
        self.y_train = self.training_set['label'].values
        
        # Load test data if needed
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']].values
            self.y_test = self.test_set['label'].values
        
    def model_fit_linear(self):
        """Initialize and fit the Linear Regression model"""
        assert self.X_train is not None and self.y_train is not None, "Training data is not initialized"
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)

    def model_fit_logistic(self):
        """Initialize and fit the Logistic Regression model"""
        assert self.X_train is not None and self.y_train is not None, "Training data is not initialized"
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        """Predict with the linear regression model and calculate metrics"""
        self.model_fit_linear()
        
        # Perform prediction
        y_pred_continuous = self.model_linear.predict(self.X_test)
        y_pred = np.where(y_pred_continuous >= 0.5, 1, 0)  # Convert continuous to binary
        
        # Calculate accuracy and classification metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)
        
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        """Predict with the logistic regression model and calculate metrics"""
        self.model_fit_logistic()
        
        # Perform prediction
        y_pred = self.model_logistic.predict(self.X_test)
        
        # Calculate accuracy and classification metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)
        
        return [accuracy, precision, recall, f1, support]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating dataset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    
    # Ensure test flag is provided to evaluate the model
    if args.perform_test:
        acc = classifier.model_predict_linear()
        print("Linear Regression Metrics: ", acc)
        
        acc = classifier.model_predict_logistic()
        print("Logistic Regression Metrics: ", acc)
    else:
        print("Perform test flag (-t) is not set.")
