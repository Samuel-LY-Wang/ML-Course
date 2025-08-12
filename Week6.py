import numpy as np

def hinge_loss_grad(x, y, a):
    if (y*a>1):
      return np.zeros(x.shape)
    return x*np.sign(a)

#2A
def softmax(z):
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))
    return ez / np.sum(ez, axis=0, keepdims=True)
print(softmax([-1,0,1]))

#2C-F
w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1], [1]])
y = np.array([[0, 1, 0]]).T
a = softmax(np.dot(w.T, x)) # probabilities
grad = np.dot(x, (a-y).T) # NLL loss gradient w.r.t weights
step_size = 0.5

def gradient_descent_step(w, grad, step_size):
    return w - step_size * grad

def relu(x):
    return np.maximum(0, x)
#multilayer neural net
# layer 1 weights, ReLU activation
w_1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w_1_bias = np.array([[-1, -1, -1, -1]]).T

# layer 2 weights, softmax activation
w_2 = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
w_2_bias = np.array([[0, 2]]).T

x1=np.array([[0.5], [0.5]])
x2=np.array([[0], [2]])
x3=np.array([[-3], [0.5]])
hiddenoutput1 = relu(np.dot(w_1.T, x1) + w_1_bias)
hiddenoutput2 = relu(np.dot(w_1.T, x2) + w_1_bias)
hiddenoutput3 = relu(np.dot(w_1.T, x3) + w_1_bias)
print(hiddenoutput1)
print(hiddenoutput2)
print(hiddenoutput3)

import numpy as np

T  = np.matrix([[0.0 , 0.1 , 0.9 , 0.0],
[0.9 , 0.1 , 0.0 , 0.0],
[0.0 , 0.0 , 0.1 , 0.9],
[0.9 , 0.0 , 0.0 , 0.1]])
g = 0.9
r = np.matrix([0, 1., 0., 2.]).reshape(4, 1)

print(np.linalg.solve(np.eye(4) - g * T, r))