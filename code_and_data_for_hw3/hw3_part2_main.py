import pdb
import numpy as np
import code_for_hw3_part2 as hw3
import csv

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------


if False: # change to True to run analysis on the auto data
    # Returns a list of dictionaries.  Keys are the column names, including mpg.
    auto_data_all = hw3.load_auto_data('code_and_data_for_hw3/auto-mpg.tsv')

    # The choice of feature processing for each feature, mpg is always raw and
    # does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
    # 'name' is not numeric and would need a different encoding.
    # features = [('cylinders', hw3.raw),
    #             ('displacement', hw3.raw),
    #             ('horsepower', hw3.raw),
    #             ('weight', hw3.raw),
    #             ('acceleration', hw3.raw),
    #             ## Drop model_year by default
    #             ## ('model_year', hw3.raw),
    #             ('origin', hw3.raw)]
    norm_features = [('cylinders', hw3.one_hot),
                ('displacement', hw3.standard),
                ('horsepower', hw3.standard),
                ('weight', hw3.standard),
                ('acceleration', hw3.standard),
                ## Drop model_year by default
                ## ('model_year', hw3.raw),
                ('origin', hw3.one_hot)]

    # Construct the standard data and label arrays
    # auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
    # print('auto data and labels shape', auto_data.shape, auto_labels.shape)
    norm_data, norm_labels = hw3.auto_data_and_labels(auto_data_all, norm_features)
    print('normalized auto data and labels shape', norm_data.shape, norm_labels.shape)

    # T5_p = hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10, 1)
    # T5_a = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10, 1)
    # print("T=1 done")
    # T10_P = hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10, 10)
    # T10_a = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10, 10)
    # print("T=10 done")
    # T50_P = hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10, 50)
    # T50_a = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10, 50)
    # print("T=50 done")
    # print(T5_p, T5_a)
    # print(T10_P, T10_a)
    # print(T50_P, T50_a)
    # print("-"*50)

    # T5_p = hw3.xval_learning_alg(hw3.perceptron, norm_data, norm_labels, 10, 1)
    # T5_a = hw3.xval_learning_alg(hw3.averaged_perceptron, norm_data, norm_labels, 10, 1)
    # print("T=1 done")
    T10_a = hw3.xval_learning_alg(hw3.averaged_perceptron, norm_data, norm_labels, 10, 10)
    # print("T=10 done")
    # T50_P = hw3.xval_learning_alg(hw3.perceptron, norm_data, norm_labels, 10, 50)
    # T50_a = hw3.xval_learning_alg(hw3.averaged_perceptron, norm_data, norm_labels, 10, 50)
    # print("T=50 done")
    # print(T5_p, T5_a)
    print(T10_a[0])
    print(T10_a[1])
    # print(T50_P, T50_a)
    exit()

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
if False: # change to true for review data (VERY SLOW)
    review_data = hw3.load_review_data('code_and_data_for_hw3/reviews.tsv')

    # Lists texts of reviews and list of labels (1 or -1)
    review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

    # The dictionary of all the words for "bag of words"
    dictionary = hw3.bag_of_words(review_texts)

    # The standard data arrays for the bag of words
    review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
    review_labels = hw3.rv(review_label_list)
    print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

    # T5_p = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, 1)
    # T5_a = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, 1)

    # T10_P = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, 10)
    word_list=list(dictionary.keys())
    # print(dictionary)
    T10_a = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, 10)
    print("Training done. Writing result to file...")
    weights=T10_a[1][0]
    print(weights)
    with open("output.csv", "w") as f:
        writer = csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(weights)):
            writer.writerow([word_list[i], weights[i]])
    print("Done writing to file")

    # T50_P = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, 50)
    # T50_a = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, 50)

    # print(T5_p, T5_a)
    print(T10_a[0])
    exit()
# print(T50_P, T50_a)


#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[9]["images"]
d1 = mnist_data_all[0]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    s, m, n=np.shape(x)
    new_arr=np.reshape(x, (s, m*n))
    return new_arr.T

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    averages = np.average(x, axis=2)
    return averages.T
    raise Exception("modify me!")


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    averages = np.average(x, axis=1)
    return averages.T
    raise Exception("modify me!")


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    samples, m, _ = np.shape(x)
    features=np.array((2, samples))
    for i in range(samples):
        sample=x[i]
        features[0][i]=np.mean(sample[:m//2])
        features[1][i]=np.mean(sample[m//2:])
    return features
    raise Exception("modify me!")

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
print(acc[0])

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

