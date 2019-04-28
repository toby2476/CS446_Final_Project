
import numpy as np


def viterbi(a,b,s,seq):
	'''
	Parameters

	a:	Transition probabilities (num_states x num_states)
	b: 	Emission probabilities (num_states x alphabet)
	s:  Starting probabilities (num_states,)
	seq: Sequence of observations 

	Variables

	v: DP Matrix (len(seq) x num_states)
	bt: Backtrace Matrix (len(seq) x num_states)

	Retvals:

	prob: P(O,Q), the joint probability for list of states and list of observations
	state_list: The list of predicted states in sequence
	'''

	v = np.zeros((len(seq),len(s)))
	bt = np.zeros_like(v)

	#FILL DP TABLE

	for t in range(len(seq)):
		for j in range(len(s)):

			#Base Case
			if t == 0:
				v[t,j] = s[j]*b[j,seq[t]]
				bt[t,j] = -1					#use -1 to denote start state, 0...num_states-1 to denote other states

			#Recursive Case
			else:
				v[t,j] = 0
				bt[t,j] = -1
				for i in range(len(s)):
					if v[t-1,i]*a[i,j]*b[j,seq[t]] > v[t,j]:
						v[t,j] = v[t-1,i]*a[i,j]*b[j,seq[t]]
						bt[t,j] = i


	#BACKTRACE

	prob = np.amax(v[-1,:])
	state_list = [int(np.argmax(v[-1,:]))]

	for t in range(len(seq)-1,-1,-1):
		state = state_list[-1]
		state_list.append(int(bt[t,state]))

	state_list.reverse()

	return prob, state_list[1:]
		
				
#Test Using Homework 4 Data


a = np.array([[0.9,0.1],[0.1,0.9]])
b = np.array([[0.25,0.25,0.25,0.25],[0.3,0.3,0.2,0.2]])
s = np.array([0.75,0.25])
seq = np.array([0,1,1,3])

p, states = viterbi(a,b,s,seq)
print("Joint probability: ",p)
print("List of states: ",states)





