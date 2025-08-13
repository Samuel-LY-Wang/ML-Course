import numpy as np

class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights


class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        return np.dot(self.W.T, A)+self.W0  # Your code (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW = np.dot(self.A, dLdZ.T)       # Your code
        self.dLdW0 = np.sum(dLdZ, axis=1, keepdims=True)     # Your code
        return np.dot(self.W, dLdZ)            # Your code: return dLdA (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
        self.W = self.W - lrate*self.dLdW           # Your code
        self.W0 = self.W0 - lrate*self.dLdW0          # Your code


class Tanh(Module):            # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):    # Uses stored self.A
        return dLdA*(np.ones(self.A.shape)-np.power(self.A,2)) # Your code: return dLdZ


class ReLU(Module):              # Layer activation
    def forward(self, Z):
        self.A = np.maximum(0, Z)            # Your code
        return self.A

    def backward(self, dLdA):    # uses stored self.A
        ReLU_grad = (self.A > 0).astype(float)  # Your code
        return dLdA * ReLU_grad                 #returns dL/dZ


class SoftMax(Module):           # Output activation
    def forward(self, Z):
        ez = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # subtracts out the max for stability
        return ez / np.sum(ez, axis=0, keepdims=True) # outputs softmax probabilities

    def backward(self, dLdZ):    # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred, axis=0)


class NLL(Module):       # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        # If Y is one-hot: get indices
        if Y.ndim > 1 and Y.shape[0] > 1:
            Y_idx = np.argmax(Y, axis=0)
        else:
            Y_idx = Y.flatten()
        # Gather predicted probabilities for correct classes
        probs = Ypred[Y_idx, np.arange(Ypred.shape[1])]
        # Compute NLL loss
        return -np.sum(np.log(probs + 1e-9))

    def backward(self):
        # Ypred must be softmax probabilities of shape (C, B)
        Ypred = self.Ypred
        Y = self.Y
        B = Ypred.shape[1]

        if Y.ndim == 2 and Y.shape[0] > 1:
            # one-hot targets shaped (C, B)
            grad = Ypred - Y                    # dL/dz = p - y
        else:
            # class indices shaped (B,) or (1, B)
            idx = Y.flatten().astype(int)
            grad = Ypred.copy()
            grad[idx, np.arange(B)] -= 1        # subtract 1 at the true class

        return grad                          # average over batch


class Sequential:
    def __init__(self, modules, loss):            # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        for it in range(iters):
            # Randomly select a data point
            idx = np.random.randint(N)
            x_i = X[:, idx:idx+1]  # shape (D, 1)
            y_i = Y[:, idx:idx+1]  # shape (C, 1) or (1, 1)
            # Forward pass
            y_pred = self.forward(x_i)
            # Compute loss
            cur_loss = self.loss.forward(y_pred, y_i)
            # Backward pass
            delta = self.loss.backward()
            self.backward(delta)
            # SGD step
            self.sgd_step(lrate)
            self.print_accuracy(it, X, Y, cur_loss)

    def forward(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                    # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):                    # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '	Acc =', acc, '	Loss =', cur_loss)