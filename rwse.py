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

def showHeatMap(matrix, title):
	plt.imshow(matrix, cmap='gray', interpolation='nearest', vmin=0)
	plt.title(title)
	plt.show()

def solveHomogeneousSystem(A, eps=1e-15):
	u, s, vh = np.linalg.svd(A)
	null_space = np.compress(s <= eps, vh, axis=0)
	return null_space.T

def testStickiness(n):
	P = transitionMatrix(n)
	d = stickinessForFlatDistribution(P)
	showHeatMap(np.reshape(d,(n,n)))

def stationaryDistribution(P):
	I = np.identity(P.shape[0])
	A = (P - I).transpose()
	b = solveHomogeneousSystem(A).transpose()
	return b / b.sum()

def testStationaryDistribution(n):
	P = transitionMatrix(n)
	x = stationaryDistribution(P)
	showHeatMap(np.reshape(x,(n,n)))

def stickinessForFlatDistribution(P):
	one = np.ones([P.shape[0],1])
	pi = stationaryDistribution(P)
	return one.transpose() - (1/pi.max()) * pi

def testStickinessSolution(n):
	I = np.identity(n**2)
	P = transitionMatrix(n)
	pi = stationaryDistribution(P)
	showHeatMap(np.reshape(pi,(n,n)), 'Stationary distribution for transition matrix P')
	d = stickinessForFlatDistribution(P)
	showHeatMap(np.reshape(d,(n,n)), 'Stickiness probabilities d for flat distribution')
	D = np.diagflat(d)
	M = np.dot(I-D,P) + D
	mu = stationaryDistribution(M)
	showHeatMap(np.reshape(mu,(n,n)), 'Distribution for transition matrix with sticking M = (I-D)P + D')

if __name__=='__main__':
	testStickinessSolution(20)
