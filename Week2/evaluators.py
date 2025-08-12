import numpy as np

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    theta, theta0=learner(data_train, labels_train)
    a = np.matmul(data_test.T,theta)+theta0
    predict_labels = np.transpose(np.sign(a))
    scores = (predict_labels == labels_test)
    return np.mean(scores)

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    total_score=0.0
    for i in range(it):
        train_data, train_labels=data_gen(n_train)
        test_data, test_labels=data_gen(n_test)
        theta, theta0=learner(train_data, train_labels)
        a=np.matmul(test_data.T,theta)+theta0
        a=np.sign(a).T
        scores=(a==test_labels)
        total_score += np.count_nonzero(scores)
    return total_score/(n_test*it)

def xval_learning_alg(learner, data, labels, k):
    data_split=np.array_split(data.T, k)
    labels_split=np.array_split(labels.flatten(), k)
    total_score=0.0
    for i in range(k):
        # test data and labels
        test_data=data_split[i]
        test_labels=labels_split[i]
        # train data and labels
        train_data=np.concatenate(data_split[:i]+data_split[i+1:],axis=0)
        train_labels=np.concatenate(labels_split[:i]+labels_split[i+1:],axis=0)
        theta, theta0=learner(train_data.T, np.array([train_labels]))
        a = np.dot(test_data,theta)+theta0
        predict_labels = np.sign(a)
        predict_labels = predict_labels.flatten()
        scores=(predict_labels == test_labels)
        total_score += np.sum(scores)/np.size(test_labels)
    #cross validation of learning algorithm
    return total_score/k