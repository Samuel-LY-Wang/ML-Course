import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    d,n=data.shape
    theta=np.zeros((d,1))
    theta0=np.zeros((1))
    for t in range(T):
        for i in range(n):
            y=labels[0,i]
            x=data[:,i]
            a=np.dot(x,theta)+theta0
            iscorrect=(np.sign(a)==np.sign(y))
            if (not iscorrect):
                theta[:,0]=theta[:,0]+y*x
                theta0 += y
        #do something
    return (theta,np.array([theta0]))

import numpy as np

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    d,n=data.shape
    ths=np.zeros((d,1))
    th0s=np.zeros((1))
    theta=np.zeros((d,1))
    theta0=np.zeros((1))
    for t in range(T):
        for i in range(n):
            y=labels[0,i]
            x=data[:,i]
            a=np.dot(x,theta)+theta0
            iscorrect=(np.sign(a)==np.sign(y))
            if (not iscorrect):
                theta[:,0]=theta[:,0]+y*x
                theta0 += y
            ths = ths+theta
            th0s = th0s+theta0
        
        #do something
    return (ths/(n*T),np.array([th0s/(n*T)]))