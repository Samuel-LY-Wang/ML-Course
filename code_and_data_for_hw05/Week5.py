import numpy as np

def num_grad(f):
    def df(x):
        g = np.zeros(x.shape)
        delta = 0.001
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            xm = f(x)
            x[i,0] = xi + delta
            xp = f(x)
            x[i,0] = xi
            g[i,0] = (xp - xm)/(2*delta)
        return g
    return df

# th = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), Y.T)

X = np.array([[1, 2], [2, 3], [3, 5], [1, 4]])
a=np.dot(X, X.T)
print(a)
print(np.linalg.inv(a))

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar

def lin_reg(x, th, th0):
    return np.dot(th.T, x) + th0

def square_loss(x, y, th, th0):
    return (y - lin_reg(x, th, th0))**2

def mean_square_loss(x, y, th, th0):
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)

def d_lin_reg_th(x, th, th0):
    return x
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th.
def d_square_loss_th(x, y, th, th0):
    return -2*(y-lin_reg(x, th, th0))*x

def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

def d_lin_reg_th0(x, th, th0):
    return np.ones((1, x.shape[1]))
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th0.
def d_square_loss_th0(x, y, th, th0):
    return -2*(y-lin_reg(x, th, th0))

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses d_square_loss_th0.
def d_mean_square_loss_th0(x, y, th, th0):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

def ridge_obj(x, y, th, th0, lam):
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2

def d_ridge_obj_th(x, y, th, th0, lam):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)+2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

def J(Xi, yi, w):
    # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
    return float(ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, yi, w):
    def f(w): return J(Xi, yi, w)
    return num_grad(f)(w)

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    d, n = X.shape
    w=w0.copy()
    ws=[]
    fs=[]
    for i in range(max_iter):
        step_size = step_size_fn(i)
        t = np.random.randint(n)
        cur_x = X[:,t].reshape(d, 1)
        cur_y = y[0,t]
        f=J(cur_x, cur_y, w)
        ws.append(w.copy())
        fs.append(f)
        if (i == max_iter-1):
            return (w, fs, ws)
        w = w-step_size*dJ(cur_x, cur_y, w)