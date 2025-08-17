"""
This is just a file for bashing out certain problems
Written in a fairly low-level and quick manner because I don't feel like getting keras.layers.Conv1D to work properly.
"""

import numpy as np

def gen_string(len, prob_of_one):
    """ Generate a random binary string of given length and probability of 1s."""
    return ''.join(['1' if np.random.rand() < prob_of_one else '0' for _ in range(len)])

p_one=0.1
len=1024
its=1000
total_loss=0
average_filter = [1/3, 1/3, 1/3]
for _ in range(its):
    s=gen_string(len, p_one)
    num_zero_blocks = sum(1 for i in range(1, len) if s[i]=='0' and s[i-1]=='1')
    if (s[0]=='0'):
        num_zero_blocks += 1
    pred = np.sum(np.convolve([int(c) for c in s], average_filter, mode='same'))-10
    total_loss += (pred - num_zero_blocks)**2
print(total_loss / its)