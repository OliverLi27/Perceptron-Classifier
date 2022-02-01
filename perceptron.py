'''
Created on Jan 25, 2022
@author: Xingchen Li
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sign(x):
    return np.where(x >= 0, 1, -1)

def load_iris(train=True):
    path = "./dataset/iris.data"    
    df= pd.read_csv(path,header=None)
    if train:
        # Row 0 to 100, column 5
        y = df.iloc[0:100, 4].values
        # Digitize the target value to iris-setosa to -1, otherwise to 1
        y = np.where(y == "Iris-setosa", -1, 1)
        # Fetch the values for rows 0 through 100, columns 1 and 3
        x = df.iloc[0:100, [1, 3]].values
        y = y[:,np.newaxis]
    else:
        # Lines 101 through 150, column 5
        y = df.iloc[100:, 4].values
        # Digitize the target value to iris-setosa to -1, otherwise to 1
        y = np.where(y == "Iris-setosa", -1, 1)
        # Fetch the values for rows 101 through 150, columns 2 and 4
        x = df.iloc[100:, [1, 3]].values
        y = y[:,np.newaxis]
    return np.concatenate((x, y), axis= -1)

def acc(y, y_pred):
    return np.sum(y == y_pred) / len(y)

class Perceptron(object):
    def __init__(self, input_dim, lr=1):
        # Initialize w,b is 0
        self.w = np.zeros(input_dim)
        self.b = np.zeros(1)
        self.lr = lr
        self.epoch = 10
        self.iterations_per_epoch = 100
        self.w_list = []
        self.b_list = []

    def train(self, train_data):
        # Random gradient descent was used for training
        # It's not random, it's going to start from scratch every time looking for the first misclassified sample
        i = 0
        count = 0
        error_nums = [] # The number of samples with classification errors in each iteration was counted
        while(i < len(train_data)):
            xi, yi = train_data[i][:2], train_data[i][2:3]
            # xi, yi = train_data[i]
            if (yi * (np.dot(self.w, xi) + self.b) <= 0):
                TP, FN, FP, TN = self.evaluate(train_data)
                error_nums.append(FN + FP)
                self.w_list.append(self.w)
                self.b_list.append(self.b)
                # Update W and B according to the gradient formula
                self.w = self.w + self.lr * yi * xi
                self.b = self.b + self.lr * yi
                print("The number of iterations: : ", count, "Misclassification point: ", i + 1, " w = ", self.w, ", b = ", self.b)
                # print("Misclassification point: ", count, " Misclassification point: ", i + 1)
                i = 0
                count = count + 1
                if (count > self.iterations_per_epoch):
                    break
                continue
            i = i + 1
        # Record w and B for each iteration
        self.w_list.append(self.w)
        self.b_list.append(self.b)
        # Calculate a list of evaluation indexes (precision,recall and F1 are all 1 in the case of linear separability)
        TP, FN, FP, TN = self.evaluate(train_data)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        f1 = (2 * precision * recall ) / (precision + recall)
        print("accuracy: ", accuracy, ", precision: ", precision, ", recall: ", recall, ", f1: ", f1)
        return error_nums

    def evaluate(self, train_data):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for item in train_data:
            # xi, yi = item
            xi, yi = item[:2], item[2:3]
            flag = sign(np.dot(self.w, xi) + self.b)
            if yi == 1 and flag == 1:
                TP = TP + 1
            if yi == 1 and flag == -1:
                FN = FN + 1
            if yi == -1 and flag == 1:
                FP = FP + 1
            if yi == -1 and flag == -1:
                TN = TN + 1
        return TP, FN, FP, TN

    def predict(self, X_test):
        return sign(X_test.dot(self.w) + self.b)

def plot_result(train_data, w_list, b_list):
    plt.xlabel("petal width")
    plt.ylabel("sepal width")
    plt.legend(loc="upper left")
    plt.axis(np.array([-1, 5, 0, 2]) )
    plt.scatter(train_data[0:50, 0], train_data[0:50, 1], color="red", marker="o", label="setosa")
    plt.scatter(train_data[50:100, 0], train_data[50:100, 1], color="blue", marker="x", label="versicolor")
    for i in range(len(b_list)):
        w1, w2 = w_list[i]
        b = b_list[i]
        x = np.linspace(-1, 5, 20)
        y = -(b + w1 * x) / (w2 + 0.0000000000001)
        plt.plot(x, y, linewidth=3)
    # plt.show() 
    plt.savefig("xxx.png")

if __name__ == "__main__":
    train_data = load_iris()
    perceptron = Perceptron(2, lr=1)
    error_nums = perceptron.train(train_data)  
    test_data = load_iris(train=False)
    X_test, y_test = test_data[:,0:2],test_data[:,2]
    y_pred = perceptron.predict(X_test)
    print("On the test set acc: ", acc(y_test, y_pred))
    plot_result(np.append(train_data, test_data, axis=0), perceptron.w_list, perceptron.b_list)

    # X = np.array([[3, 3], [4, 3], [1, 1]])
    # y = np.array([1, 1, -1])
    # train_data = [(X[i], y[i]) for i in range(len(y))]
    # perceptron = Perceptron(2, lr=1)
    # error_nums = perceptron.train(train_data)  
    # test_data = load_iris(train=False)
    # X_test, y_test = test_data[:,0:2],test_data[:,2]
    # y_pred = perceptron.predict(X_test)
    # print(acc(y_test, y_pred))
    # plot_result1(train_data, perceptron.w_list, perceptron.b_list)