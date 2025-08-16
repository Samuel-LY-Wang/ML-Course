import numpy as np
import math

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


class sgd_Sequential:
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

# This OPTIONAL problem has you extend your homework 7
# implementation for building neural networks.  
# PLEASE COPY IN YOUR CODE FROM HOMEWORK 7 TO COMPLEMENT THE CLASSES GIVEN HERE

# Recall that your implementation from homework 7 included the following classes:
    # Module, Linear, Tanh, ReLU, SoftMax, NLL and Sequential

######################################################################
# OPTIONAL: Problem 2A) - Mini-batch GD
######################################################################

class mgd_Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:,indices]  # Your code
            Y = Y[:,indices]  # Your code

            for j in range(math.floor(N/K)):
                if num_updates >= iters: break

                # Implement the main part of mini_gd here
                # Your code
                x=X[:,K*j:K*(j+1)]
                y=Y[:,K*j:K*(j+1)]
                y_pred = self.forward(x)
                cur_loss = self.loss.forward(y_pred, y)
                if notif_each:
                    print(f"Iteration {num_updates}: loss = {cur_loss}")
                delta = self.loss.backward()
                print(cur_loss, delta)
                self.backward(delta)
                self.step(lrate)
                
                num_updates += 1

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def step(self, lrate):    
        for m in self.modules: m.sgd_step(lrate)
          
######################################################################
# OPTIONAL: Problem 2B) - BatchNorm
######################################################################

class Module:
    def step(self, lrate): pass  # For modules w/o weights

class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(A, axis=1, keepdims=True)  # Your Code
        self.vars = np.var(A, axis=1, keepdims=True)  # Your Code
        try:
            self.mus_r = 0.9 * self.mus_r + 0.1 * self.mus
            self.vars_r = 0.9 * self.vars_r + 0.1 * self.vars
        except AttributeError:
            self.mus_r = self.mus.copy()
            self.vars_r = self.vars.copy()

        # Normalize inputs using their mean and standard deviation
        self.norm = np.divide(A-self.mus, np.sqrt(self.vars+self.eps))  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.G*self.norm + self.B  # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B -= lrate*self.dLdB  # Your Code
        self.G -= lrate*self.dLdG # Your Code


######################################################################
# Tests
######################################################################
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)
  
def for_softmax(y):
    return np.vstack([1-y, y])
  
# For problem 1.1: builds a simple model and trains it for 3 iters on a simple dataset
# Verifies the final weights of the model
def mini_gd_test():
    np.random.seed(0)
    nn = mgd_Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 3, lrate=0.005, K=1)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]

# For problem 1.2: builds a simple model with a BatchNorm layer
# Trains it for 1 iter on a simple dataset and verifies, for the BatchNorm module (in order): 
# The final shifts and scaling factors (self.B and self.G)
# The final running means and variances (self.mus_r and self.vars_r)
# The final 'self.norm' value
def batch_norm_test():
    np.random.seed(0)
    nn = mgd_Sequential([Linear(2,3), ReLU(), Linear(3,2), BatchNorm(2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 1, lrate=0.005, K=2)
    return [np.vstack([nn.modules[3].B, nn.modules[3].G]).tolist(), \
    np.vstack([nn.modules[3].mus_r, nn.modules[3].vars_r]).tolist(), nn.modules[3].norm.tolist()]
print(batch_norm_test())