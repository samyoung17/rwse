import numpy as np
import matplotlib.pyplot as plt

e_right = (1,0)
e_left = (-1,0)
e_down = (0,1)
e_up = (0,-1)

def transitionProbabilities(n):
	def w(u):
		if tuple(u) == (0,0):
			return {e_right: .5, e_down: .5}
		elif tuple(u) == (n-1,0):
			return {e_left: .5, e_down: .5}
		elif tuple(u) == (0,n-1):
			return {e_right: .5, e_up: .5}
		elif tuple(u) == (n-1,n-1):
			return {e_left: .5, e_up: .5}
		elif u[0] == 0:
			return {e_right: 1.0/3, e_up: 1.0/3, e_down: 1.0/3}
		elif u[0] == n-1:
			return {e_left: 1.0/3, e_up: 1.0/3, e_down: 1.0/3}
		elif u[1] == 0:
			return {e_right: 1.0/3, e_left: 1.0/3, e_down: 1.0/3}
		elif u[1] == n-1:
			return {e_right: 1.0/3, e_left: 1.0/3, e_up: 1.0/3}
		else:
			return {e_right: .26, e_left: .24, e_up: .25, e_down: .25}
	return w

def p(w,u,v):
	if tuple(v-u) in w(u).keys():
		return w(u)[tuple(v-u)]
	else:
		return 0

def latticeIndices(n):
	V = []
	for i in range(n):
		for j in range(n):
			V.append(np.array([j,i]))
	return V

def transitionMatrix(n):
	V = latticeIndices(n)
	P = np.zeros([n**2,n**2])
	w = transitionProbabilities(n)
	for i, u in enumerate(V):
		for j, v in enumerate(V):
			P[i,j] = p(w,u,v)
	return P

def showHeatMap(matrix):
	plt.imshow(matrix, cmap='hot', interpolation='nearest')
	plt.show()

def solveSingularSystem(A,b,y):
	tolerance = 1E-10
	I = np.identity(len(b))
	Adag = np.linalg.pinv(A)
	diff = np.dot(A,np.dot(Adag,b)) - b
	if np.max(np.abs(diff)) > tolerance:
		raise ValueError('System has no solution')
	return np.dot(Adag,b) + np.dot(I - np.dot(Adag,A), y)


def calculateStickinessForFlatDistribution(n):
	s = 0.6
	P = transitionMatrix(n)
	I = np.identity(n**2)
	one = np.ones([1,n**2])
	p = np.dot(one, P)
	A = (I-P).transpose()
	b = (one - p).transpose()
	y = s * one.transpose()
	d = solveSingularSystem(A,b,y).transpose()
	return d

def testStickiness(n):
	d = calculateStickinessForFlatDistribution(n)
	showHeatMap(np.reshape(d,(n,n)))
