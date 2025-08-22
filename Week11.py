import numpy as np
from code_for_lab11.sm import *
from code_for_lab11.util import *
from code_for_lab11.code_for_lab11 import *

class RNN:
    weight_scale = .0001
    def __init__(self, input_dim, hidden_dim, output_dim, loss_fn, f2, dloss_f2, step_size=0.1,
                 f1 = tanh, df1 = tanh_gradient, init_state = None,
                 Wsx = None, Wss = None, Wo = None, Wss0 = None, Wo0 = None,
                 adam = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.dloss_f2 = dloss_f2
        self.step_size = step_size
        self.f1 = f1
        self.f2 = f2
        self.df1 = df1
        self.adam = adam
        self.init_state = init_state if init_state is not None else \
                                         np.zeros((self.hidden_dim, 1))
        self.hidden_state = self.init_state
        self.t = 0

        # Initialize weight matrices
        self.Wsx = Wsx if Wsx is not None \
                    else np.random.random((hidden_dim, input_dim)) * self.weight_scale
        self.Wss = Wss if Wss is not None \
                       else np.random.random((hidden_dim, hidden_dim)) * self.weight_scale
        self.Wo = Wo if Wo is not None \
                     else np.random.random((output_dim, hidden_dim)) * self.weight_scale
        self.Wss0 = Wss0 if Wss0 is not None \
                         else np.random.random((hidden_dim, 1)) * self.weight_scale
        self.Wo0 = Wo0 if Wo0 is not None \
                       else np.random.random((output_dim, 1)) * self.weight_scale

        # Initialization for ADAM
        if adam:
            self.dLdWsx_sq = np.zeros_like(self.Wsx)
            self.dLdWo_sq = np.zeros_like(self.Wo)
            self.dLdWss_sq = np.zeros_like(self.Wss)
            self.dLdWo0_sq = np.zeros_like(self.Wo0)
            self.dLdWss0_sq = np.zeros_like(self.Wss0)

            self.dLdWsx_m = np.zeros_like(self.Wsx)
            self.dLdWo_m = np.zeros_like(self.Wo)
            self.dLdWss_m = np.zeros_like(self.Wss)
            self.dLdWo0_m = np.zeros_like(self.Wo0)
            self.dLdWss0_m = np.zeros_like(self.Wss0)


    # Just one step of forward propagation.  x and y are for a single time step
    # Depends on self.hidden_state and reassigns it
    # Returns predicted output, loss on this output, and dLoss_dz2
    def forward_propagation(self, x):
        new_state = self.f1(np.dot(self.Wsx, x) +
                            np.dot(self.Wss, self.hidden_state) + self.Wss0)
        z2 = np.dot(self.Wo, new_state) + self.Wo0
        p = self.f2(z2)
        self.hidden_state = new_state
        return p

    def forward_prop_loss(self, x, y):
        p = self.forward_propagation(x)
        loss = self.loss_fn(p, y)
        dL_dz2 = self.dloss_f2(p, y)
        return p, loss, dL_dz2

    def b(self, xs, dL_dz2, states):
        dC = np.zeros_like(self.Wsx)
        dB = np.zeros_like(self.Wss)
        dA = np.zeros_like(self.Wo)
        dB0 = np.zeros_like(self.Wss0)
        dA0 = np.zeros_like(self.Wo0)
        dLfuture_dst = np.zeros((self.hidden_dim, 1))
        k = xs.shape[1]
        for t in range(k-1, -1, -1):
            xt = xs[:, t:t+1]
            st = states[:, t:t+1]
            st_minus_1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
            dL_dz2t = dL_dz2[:, t:t+1]
            dL_dA = np.dot(dL_dz2t, st.T)
            dL_dA0 = dL_dz2t
            dLtfuture_dst = np.dot(self.Wo.T, dL_dz2t) + dLfuture_dst
            dLtfuture_dz1t = dLtfuture_dst * self.df1(st)
            dLtfuture_dB = np.dot(dLtfuture_dz1t, st_minus_1.T)
            dLtfuture_dB0 = dLtfuture_dz1t
            dLtfuture_dC = np.dot(dLtfuture_dz1t, xt.T)
            dLfuture_dst = np.dot(self.Wss.T, dLtfuture_dz1t)
            dC += dLtfuture_dC
            dB += dLtfuture_dB
            dB0 += dLtfuture_dB0
            dA += dL_dA
            dA0 += dL_dA0
        return dC, dB, dA, dB0, dA0

    # With adagrad
    def sgd_step(self, xs, dLdz2s, states,
                 gamma1 = 0.9, gamma2 = 0.999, fudge = 1.0e-8):
        dWsx, dWss, dWo, dWss0, dWo0 = self.b(xs, dLdz2s, states)

        self.t += 1

        if self.adam:
            self.dLdWsx_m = gamma1 * self.dLdWsx_m + (1 - gamma1) * dWsx
            self.dLdWo_m = gamma1 * self.dLdWo_m + (1 - gamma1) * dWo
            self.dLdWss_m = gamma1 * self.dLdWss_m + (1 - gamma1) * dWss
            self.dLdWo0_m = gamma1 * self.dLdWo0_m + (1 - gamma1) * dWo0
            self.dLdWss0_m = gamma1 * self.dLdWss0_m + (1 - gamma1) * dWss0

            self.dLdWsx_sq = gamma2 * self.dLdWsx_sq + (1 - gamma2) * dWsx ** 2
            self.dLdWo_sq = gamma2 * self.dLdWo_sq + (1 - gamma2) * dWo ** 2
            self.dLdWss_sq = gamma2 * self.dLdWss_sq + (1 - gamma2) * dWss ** 2
            self.dLdWo0_sq = gamma2 * self.dLdWo0_sq + (1 - gamma2) * dWo0 ** 2
            self.dLdWss0_sq = gamma2 * self.dLdWss0_sq + (1 - gamma2) * dWss0 ** 2

            # Correct for bias
            dLdWsx_mh = self.dLdWsx_m / (1 - gamma1**self.t)
            dLdWo_mh = self.dLdWo_m / (1 - gamma1**self.t)
            dLdWss_mh = self.dLdWss_m / (1 - gamma1**self.t)
            dLdWo0_mh = self.dLdWo0_m / (1 - gamma1**self.t)
            dLdWss0_mh = self.dLdWss0_m / (1 - gamma1**self.t)

            dLdWsx_sqh = self.dLdWsx_sq / (1 - gamma2**self.t)
            dLdWo_sqh = self.dLdWo_sq / (1 - gamma2**self.t)
            dLdWss_sqh = self.dLdWss_sq / (1 - gamma2**self.t)
            dLdWo0_sqh =  self.dLdWo0_sq / (1 - gamma2**self.t)
            dLdWss0_sqh =  self.dLdWss0_sq / (1 - gamma2**self.t)

            self.Wsx -= self.step_size * (dLdWsx_mh /
                                          (fudge + np.sqrt(dLdWsx_sqh)))
            self.Wss -= self.step_size * (dLdWss_mh /
                                          (fudge + np.sqrt(dLdWss_sqh)))
            self.Wo -= self.step_size * (dLdWo_mh /
                                         (fudge + np.sqrt(dLdWo_sqh)))
            self.Wss0 -= self.step_size * (dLdWss0_mh /
                                           (fudge + np.sqrt(dLdWss0_sqh)))
            self.Wo0 -= self.step_size * (dLdWo0_mh /
                                          (fudge + np.sqrt(dLdWo0_sqh)))
        else:
            self.Wsx -= self.step_size * dWsx
            self.Wss -= self.step_size * dWss
            self.Wo -= self.step_size * dWo
            self.Wss0 -= self.step_size * dWss0
            self.Wo0 -= self.step_size * dWo0

    def reset_hidden_state(self):
        self.hidden_state = self.init_state

    def forward_seq(self, x, y):
        k = x.shape[1]
        dLdZ2s = np.zeros((self.output_dim, k))
        states = np.zeros((self.hidden_dim, k))
        train_error = 0.0
        self.reset_hidden_state()
        for j in range(k):
            p, loss, dLdZ2 = self.forward_prop_loss(x[:, j:j+1], y[:, j:j+1])
            dLdZ2s[:, j:j+1] = dLdZ2
            states[:, j:j+1] = self.hidden_state
            train_error += loss
        return train_error/k, dLdZ2s, states

    # For now, training_seqs will be a list of pairs of np arrays.
    # First will be l x k second n x k where k is the sequence length
    # and can be different for each pair
    def train_seq_to_seq(self, training_seqs, steps = 100000,
                         print_interval = None):
        if print_interval is None: print_interval = int(steps / 10)
        num_seqs = len(training_seqs)
        total_train_err = 0
        self.t = 0
        iters = 1
        for step in range(steps):
            i = np.random.randint(num_seqs)
            x, y = training_seqs[i]
            avg_seq_train_error, dLdZ2s, states = self.forward_seq(x, y)

            # Check the gradient computation against the numerical grad.
            # grads = self.b(x, dLdZ2s, states)
            # grads_n = self.num_grad(lambda : forward_seq(x, y, dLdZ2s,
            # states)[0])
            # compare_grads(grads, grads_n)

            self.sgd_step(x, dLdZ2s, states)
            total_train_err += avg_seq_train_error
            if (step % print_interval) == 0 and step > 0:
                print('%d/10: training error'%iters, total_train_err / print_interval, flush=True)
                total_train_err = 0
                iters += 1

    def num_grad(self, f, delta=0.001):
        out = []
        for W in (self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0):
            Wn = np.zeros(W.shape)
            out.append(Wn)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    wi = W[i,j]
                    W[i,j] = wi - delta
                    fxm = f()
                    W[i,j] = wi + delta
                    fxp = f()
                    W[i,j] = wi
                    Wn[i,j] = (fxp - fxm)/(2*delta)
        return out

    # Return a state machine made out of these weights
    def sm(self):
        return sm.RNN(self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0,
                      self.f1, self.f2)

    # Assume that input and output are same dimension
    def gen_seq(self, init_sym, seq_len, codec):
        assert self.input_dim == self.output_dim
        assert self.f2 == softmax
        result  = []
        self.reset_hidden_state()
        x = codec.encode(init_sym)
        for _ in range(seq_len):
            p = self.forward_propagation(x)
            x = np.array([np.random.multinomial(1, p.T[0])]).T
            if codec.decode(x) == '.':
                break
            result.append(codec.decode(x))
        return result

    def gen_seq_interactive(self, codec, seq_len = 100, maximize = True):
        self.reset_hidden_state()
        start = '.' + input('Starting string: ')
        result = start[1:]
        for c in start:
            p = self.forward_propagation(codec.encode(c))
        for _ in range(seq_len):
            c = codec.decode_max(p)
            #if c in ['.', '\n', ' ']:  break
            if c in ['.', '\n']:  break
            result = result + c
            x = codec.encode(c)
            p = self.forward_propagation(x)
        print(result)
        return result
    
    def gen_seq_interactive_top5(self, codec, seq_len = 100, maximize = True):
        self.reset_hidden_state()
        start = '.' + input('Starting string: ')
        result = start[1:]
        for c in start:
            p = self.forward_propagation(codec.encode(c))
        
        while True:
            c = codec.decode_max(p)
            #if c in ['.', '\n', ' '] :  break
            if c in ['.', '\n']:  break
            c_top5 = codec.decode_top5(p)
            #print("The argmax is :", c)
            print("We recommend that you type one of the top 5 most frequent alphabets that follow '" + str(result) + "' : ", c_top5)
            next_character = input("Next character after '" + str(result) + "' : ")
            result = result + next_character
            x = codec.encode(next_character)
            p = self.forward_propagation(x)
        print("Your final result is : ", result)
        return result
    

    # NEW FUNCTION FOR BPTT
    # Back propgation through time
    # xs is matrix of inputs: l by k
    # dLdz2 is matrix of output errors:  1 by k
    # states is matrix of state values: m by k
    def bptt(self, xs, dLtdz2, states):
        dWsx = np.zeros_like(self.Wsx)
        dWss = np.zeros_like(self.Wss)
        dWo = np.zeros_like(self.Wo)
        dWss0 = np.zeros_like(self.Wss0)
        dWo0 = np.zeros_like(self.Wo0)
        # Derivative of future loss (from t+1 forward) wrt state at time t
        # initially 0;  will pass "back" through iterations
        dFtdst = np.zeros((self.hidden_dim, 1))
        k = xs.shape[1]
        # Technically we are considering time steps 1..k, but we need
        # to index into our xs and states with indices 0..k-1
        for t in range(k-1, -1, -1):
            # Get relevant quantities
            xt = xs[:, t:t+1]
            st = states[:, t:t+1]
            stm1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
            dLtdz2t = dLtdz2[:, t:t+1]
            # Compute gradients step by step
            # ==> Use self.df1(st) to get dfdz1;
            # ==> Use self.Wo, self.Wss, etc. for weight matrices
            # derivative of loss at time t wrt state at time t
            dLtdst = np.dot(self.Wo.T, dLtdz2t)        # Your code
            # derivatives of loss from t forward
            dFtm1dst = dLtdst+dFtdst            # Your code
            dFtm1dz1t = dFtm1dst           # Your code
            dFtm1dstm1 = np.dot(self.Wss.T, dFtm1dz1t)          # Your code
            # gradients wrt weights
            dLtdWo = np.dot(dLtdz2t, st.T)              # Your code
            dLtdWo0 = dLtdz2t             # Your code
            dFtm1dWss = np.dot(dFtm1dz1t, stm1.T)           # Your code
            dFtm1dWss0 = dFtm1dz1t          # Your code
            dFtm1dWsx = np.dot(dFtm1dz1t, xt.T)           # Your code
            # Accumulate updates to weights
            dWsx += dFtm1dWsx
            dWss += dFtm1dWss
            dWss0 += dFtm1dWss0
            dWo += dLtdWo
            dWo0 += dLtdWo0
            # pass delta "back" to next iteration
            dFtdst = dFtm1dstm1
        return dWsx, dWss, dWo, dWss0, dWo0