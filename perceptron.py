import sys
import numpy as np
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self, learning_rate, max_epochs):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def fit(self, X_train, y_train):
        self.weights = np.zeros(X_train.shape[1])
        self.bias = 0
        
        for epoch in range(self.max_epochs):
            for x, y in zip(X_train, y_train):
                # Calculate predicted value
                pred = np.dot(self.weights, x) + self.bias
                y_pred = np.where(pred > 0, 1, -1)
                
                # Update weights and bias if prediction is incorrect
                if y != y_pred:
                    self.weights += self.learning_rate * y * x
                    self.bias += self.learning_rate * y
        
    def predict(self, X_test):
        pred = np.dot(X_test, self.weights) + self.bias
        y_pred = np.where(pred > 0, 1, -1)
        return y_pred

# print confusion matrix
def print_confusion_matrix(predicted, result):
    # print list of classes
    print("classes: {}".format(np.unique(result)))
    # print matrix
    cm = confusion_matrix(result, predicted)
    print("Confusion matrix:")
    for row in cm:
        print(row)
        
# calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    print("accuracy: {:.2f}%".format(accuracy))
    return accuracy

# Read command line arguments
params = sys.argv

# Load data from files
train_data = np.loadtxt(params[1], delimiter=",")
test_data = np.loadtxt(params[2], delimiter=",")

# Split data into features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Create and fit perceptron model
perceptron = Perceptron(learning_rate=float(params[3]), max_epochs=int(params[4]))
perceptron.fit(X_train, y_train)

# Make predictions on the test data
y_pred = perceptron.predict(X_test)

# Print the confusion matrix
print_confusion_matrix(y_pred, y_test)

# Calculate and print the accuracy
acc = accuracy(y_test, y_pred)
print("Average accuracy: {:.2f}%".format(acc))

