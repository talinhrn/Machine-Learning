import numpy as np
import pandas as pd


X_train = np.genfromtxt("20newsgroup_words_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("20newsgroup_labels_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("20newsgroup_words_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("20newsgroup_labels_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    classes = {}
    totsamples = len(y)
    
    for label in y:
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1
    
    unique_classes = sorted(classes.keys())
    counts = [classes[label] for label in unique_classes]
    class_priors = np.array(counts) / totsamples
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    # your implementation starts below
    
    alpha = 0.2  
    D = X.shape[1]  
    alphaD = alpha * D  
    K = len(class_priors)  
    
    classes = {}

    for label in y:
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1

    unique_classes = sorted(classes.keys())
    counts = [classes[label] for label in unique_classes]
    Nc = np.array(counts)
    
    P = np.zeros((K, D))
    
    for c in range(K):  
        class_data = X[y == c + 1]  
        class_word_counts = np.sum(class_data, axis=0)  
        P[c] = (class_word_counts + alpha) / (Nc[c] + alphaD)
    
    # your implementation ends above
    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    # your implementation starts below
    
    K = P.shape[0]  
    N, D = X.shape 
    
    score_values = np.zeros((N, K))
    
    for i in range(K):
     for j in range(D):
         if P[i, j] < 1e-10:
             P[i, j] = 1e-10
         elif P[i, j] > 1 - 1e-10:
             P[i, j] = 1 - 1e-10
    for c in range(K): 
        log_likelihood = np.sum(X * np.log(P[c]) + (1 - X) * np.log(1 - P[c]), axis=1)
        log_prior = np.log(class_priors[c])
        score_values[:, c] = log_likelihood + log_prior
    
    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)


# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    K = np.max(y_truth)  
    predict = np.argmax(scores, axis=1)  
    confusion_matrix = np.zeros((K, K), dtype=int)
    
    for i in range(len(y_truth)):
        confusion_matrix[y_truth[i] - 1, predict[i]] += 1
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))


