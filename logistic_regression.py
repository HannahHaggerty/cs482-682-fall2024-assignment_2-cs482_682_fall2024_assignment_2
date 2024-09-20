import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
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


    def read_csv(self, dataset_num) :
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        
        elif dataset_num == '2':
            train_dataset_file ='train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        
        else:
            print("unsupported dataset number")
            return

        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']].values
        self.y_train =  self.training_set ['label'].values

        if self.perform_test:

            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set [['exam_score_1', 'exam_score_2']].values
            self.y_test = self.test_set['label'].values
        
    def model_fit_linear(self):
        assert self.X_train is not None and self.y_train is not None, "Training data is not initialized" 
        self.model_linear = LinearRegression()

        self.model_linear.fit(self.X_train, self.y_train)
 
    def model_fit_logistic(self):
        assert self.X_train is not None and self.y_train is not None, "Training data is not initialized"
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        self.model_fit_linear()
        y_pred_continuous = self.model_linear.predict(self.X_test)
        y_pred = np.where(y_pred_continuous >= 0.5, 1, 0)#Convert continuous to binary
        accuracy = accuracy_score(self.y_test, y_pred) #metrics
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)
        return [accuracy, precision, recall, f1, support] #return all the statss

 
    def model_predict_logistic(self):
        self.model_fit_logistic()
        y_pred = self.model_logistic.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)
        return [accuracy, precision, recall, f1, support]

    def print_metrics(self, model_name, metrics):
        accuracy, precision, recall, f1, support =  metrics
        print(f"\n{model_name} Metrics:")

        print(f"Accuracy: {accuracy:.2f}")
        for i in range (len(precision)):
            print(f"Class {i}:")
            print(f"  Precision: {precision[i]:.2f}")
            print(f"  Recall:    {recall[i]:.2f}")
            print(f"  F1 Score:  {f1[i]:.2f}")
            print(f"  Support:   {support[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic and Linear Regression')
    parser.add_argument('-d', '--dataset_num', type=str, default="1", choices=["1", "2"], help='Dataset number (1 or 2)')
    parser.add_argument('-t', '--perform_test', action='store_true', help='Flag to perform testing and output results')
    args = parser.parse_args()
    
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    
    acc = classifier.model_predict_linear()
    acc = classifier.model_predict_logistic()
    if args.perform_test:
        linear_metrics = classifier.model_predict_linear() #Perform linear regression predictions and print metrics
        classifier.print_metrics("Linear Regression", linear_metrics)
        

        logistic_metrics = classifier.model_predict_logistic()# Perform logistic regression predictions and print metrics

        classifier.print_metrics("Logistic Regression", logistic_metrics)
    else:
        print("Perform test flag (-t) is not set.")