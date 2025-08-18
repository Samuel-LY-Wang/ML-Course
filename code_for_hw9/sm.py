import numpy as np

class SM:
    start_state = None

    def transition_fn(self, s, i):
        '''s:       the current state
           i:       the given input
           returns: the next state'''
        raise NotImplementedError
    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
        state=self.start_state
        output_seq=[]
        for inp in input_seq:
            state=self.transition_fn(state, inp)
            output_seq.append(self.output_fn(state))
        return output_seq
            
class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, i):
        return s + i

    def output_fn(self, s):
        return s
    
class Binary_Addition(SM):
    start_state = (0,0)

    def transition_fn(self, s, x):
        state=s[0]
        state=state+x[0]+x[1]
        return (state//2, state%2)

    def output_fn(self, s):
        return s[1]
    
class Reverser(SM):
    start_state = [False, []]

    def transition_fn(self, s, x):
        new_s=s.copy()
        if (x=='end'):
            new_s[0]=True
        elif (s[0]):
            pass
        else:
            new_s[1].append(x)
        return new_s

    def output_fn(self, s):
        if (not s[0]):
            return None
        if (len(s[1])==0):
            return None
        temp=s[1][-1]
        s[1].pop(-1)
        return temp

class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
        self.start_state=np.zeros((self.Wss.shape[1], 1))
    def transition_fn(self, s, x):
        new_s = np.dot(self.Wss, s)+np.dot(self.Wsx, x)+self.Wss_0
        return self.f1(new_s)
    def output_fn(self, s):
        out=np.dot(self.Wo, s)+self.Wo_0
        return self.f2(out)

# example RNN to compute the sign of the sum of inputs
Wsx =    np.array([[1000]])
Wss =    np.array([[1]])
Wo =     np.array([[1]])
Wss_0 =  np.array([[0]])
Wo_0 =   np.array([[0]])
f1 =     lambda x: x
f2 =     np.tanh
# acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)

# example more complex RNN
Wsx =    np.array([[1], [0], [0]])
Wss =    np.array([[0,0,0], [1,0,0], [0,1,0]])# Your code here
Wo =     np.array([[1,-2,3]])# Your code here
Wss_0 =  np.array([[0], [0], [0]])# Your code here
Wo_0 =   np.array([[0]])# Your code here
f1 =     lambda x: x# Your code here
f2 =     lambda x: x# Your code here
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)

# programmatically compute a few iterations of an MDP
Ab=np.matrix([[0, 0.81, 0.09, 0], [0.81, 0.09, 0, 0], [0, 0, 0.09, 0.81], [0.81, 0, 0, 0.09]])
Ac=np.matrix([[0, 0.09, 0.81, 0], [0.81, 0.09, 0, 0], [0, 0, 0.09, 0.81], [0.81, 0, 0, 0.09]])
base=np.matrix([0,1,0,2]).T
b1=np.dot(Ab, base)+base
c1=np.dot(Ac, base)+base
bc1=np.maximum(b1, c1)
b2=np.dot(Ab, bc1)+base
c2=np.dot(Ac, bc1)+base
bc2=np.maximum(b2, c2)
print(bc2)