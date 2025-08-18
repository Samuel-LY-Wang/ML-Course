import pdb
from dist import *
from util import *
import random

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, max_iters=1000):
    new_q=q.copy()
    # Your code here
    gamma=mdp.discount_factor
    rf=mdp.reward_fn
    T=mdp.transition_model
    states=mdp.states
    actions=mdp.actions
    for _ in range(max_iters):
        prev_q=new_q.copy()
        for s in states:
            for a in actions:
                distrib=T(s,a)
                sumofstates=0
                for s1 in states:
                    max_next_q = max([prev_q.get(s1, a1) for a1 in actions])
                    sumofstates += distrib.prob(s1) * max_next_q
                new_v=rf(s, a)+gamma*sumofstates
                new_q.set(s, a, new_v)
        terminate=True
        for s in states:
            for a in actions:
                if (new_q.get(s, a)-prev_q.get(s, a)>eps):
                    terminate=False
        if terminate:
            break
    return new_q
        

# Compute the q value of action a in state s with horizon h, using
# expectimax
def q_em(mdp, s, a, h):
    if h == 0:
        return 0
    rf = mdp.reward_fn
    T = mdp.transition_model
    gamma = mdp.discount_factor
    actions = mdp.actions
    distrib = T(s, a)
    expected = 0
    for s1 in mdp.states:
        prob = distrib.prob(s1)
        if prob > 0:
            # Expectimax: take max over actions at next state
            future = max([q_em(mdp, s1, a1, h-1) for a1 in actions])
            expected += prob * future
    return rf(s, a) + gamma * expected

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    actions=q.actions
    max_V=-100000000
    for a in actions:
        max_V=max(max_V, q.get(s, a))
    return max_V

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    actions=q.actions
    max_V=-100000000
    max_act=None
    for a in actions:
        V=q.get(s,a)
        if (V >= max_V):
            max_V=V
            max_act=a
    return a
    pass


def epsilon_greedy(q, s, eps = 0.5):
    """ Returns an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        dist=uniform_dist(q.actions)
        return dist.draw()
    else:
        return greedy(q, s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]

def tiny_reward(s, a):
    # Reward function
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0

def tiny_transition(s, a):
    # Transition function
    if s == 0:
        if a == 'b':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.9})
    
def test_value_iteration():
    tiny = MDP([0, 1, 2, 3], ['b', 'c'], tiny_transition, tiny_reward, 0.9)
    q = TabularQ(tiny.states, tiny.actions)
    qvi = value_iteration(tiny, q, eps=0.1, max_iters=100)
    expected = dict([((2, 'b'), 5.962924188028282),
                     ((1, 'c'), 5.6957634856549095),
                     ((1, 'b'), 5.6957634856549095),
                     ((0, 'b'), 5.072814297918393),
                     ((0, 'c'), 5.262109602844769),
                     ((3, 'b'), 6.794664584556008),
                     ((3, 'c'), 6.794664584556008),
                     ((2, 'c'), 5.962924188028282)])
    for k in qvi.q:
        print("k=%s, expected=%s, got=%s" % (k, expected[k], qvi.q[k]))      

test_value_iteration()