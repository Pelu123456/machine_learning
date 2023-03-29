import sys
import numpy as np
from sklearn.metrics import confusion_matrix 

# euclidean distances function

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

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

# KNN

def knn(train_set, test_set, k):
    predictions = []
    # loop over all test instances
    cnt = 0 
    for test_instance in test_set:
        cnt += 1
        print("test instance: {}".format(cnt))
        distances = []
        # loop over all train instances
        for train_instance in train_set:
            dist = distance(test_instance[:-1], train_instance[:-1])
            distances.append((dist, train_instance[-1]))
        distances.sort()
        neighbors = [x[1] for x in distances[:k]]
        prediction = max(set(neighbors), key=neighbors.count)
        predictions.append(prediction)
    accuracy(test_set[:, -1], predictions)
    print_confusion_matrix(predictions, test_set[:, -1])
    return predictions

params = sys.argv

train_data = np.loadtxt(params[1], delimiter=",")
test_data = np.loadtxt(params[2], delimiter=",")

knn(train_data, test_data, int(params[3]))

