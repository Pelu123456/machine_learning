import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 

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

params = sys.argv

train_data = np.loadtxt(params[1], delimiter=",")
test_data = np.loadtxt(params[2], delimiter=",")

# separate features and labels
X_train = train_data[:,:-1]
y_train = train_data[:,-1]
X_test = test_data[:,:-1]
y_test = test_data[:,-1]

# train the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# make predictions on the test data
y_pred = clf.predict(X_test)

# print the confusion matrix
print_confusion_matrix(y_pred, y_test)

# calculate and print the accuracy
acc = accuracy(y_test, y_pred)
print("Average accuracy: {:.2f}%".format(acc))

