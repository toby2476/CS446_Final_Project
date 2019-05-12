import numpy as np

'''
Parameters
	a: Transition probabilities (num_states x num_states)
	b: Emission probabilities (num_states x alphabet)
	pi: Starting probabilities (num_states,)
	seq: Sequence of observations
'''

#the forward algorithm
def forward(a, b, pi, seq):
	T = len(obs_seq)
    N = a.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi*b[:,seq[0]]
    for t in range(1, T):
        alpha[t] = alpha[t-1].dot(a) * b[:, seq[t]]
    return alpha

#uses the forward algorithm
 def likelihood(a, b, pi, seq):
    # returns log P(Y  \mid  model)
    # using the forward part of the forward-backward algorithm
    return  forward(a, b, pi, seq)[-1].sum()

#the backward algorithm
def backward(a, b, pi, seq):
    N = a.shape[0]
    T = len(seq)

    beta = np.zeros((N,T))
    beta[:,-1:] = 1

    for t in reversed(range(T-1)):
        for n in range(N):
            beta[n,t] = np.sum(beta[:,t+1] * a[n,:] * b[:, seq[t+1]])

    return beta

#combining all the previous algorithms
def forward_backward(a, b, pi, seq):
    alpha = forward(a, b,pi, seq)
    beta  = backward(a, b, pi, seq)
    obs_prob = likelihood(a, b, pi, seq)
    return (np.multiply(alpha,beta.T) / obs_prob)
