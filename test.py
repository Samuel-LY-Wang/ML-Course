import numpy as np

print(np.__version__)

# x = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
#                   [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
# y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])

# d, n = x.shape
# print(n)
# t = np.random.randint(n)
# print(x[:,t].reshape(d, 1), y[:,t])

test = np.array([[1, 2, 3], [4, 5, 6]])
print(np.tanh(test))