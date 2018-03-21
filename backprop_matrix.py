import operator 
import math
import numpy as np


def squash(value,function):
	if function == "sigmoid":
		return sigmoid(value)
	else:
		return None
		


class MLTools(object):

	#sigmoid squash function
	def sigmoid(value):
		exp = math.exp(-value)
		return 1/(1+exp)

	def sigmoidDerivative(value):
		return (value)*(1-value)

	@staticmethod
	def squaredError(expected,actual):
		total = 0
		for e,a in zip(expected,actual):
			total += (.5) * ((e-a)**2)  #Σ (1/2 (expected−actual)^2)
		return total


	#Dictionary containing refs to functions
	squash = {"sigmoid":sigmoid}

	#Dictionary containing refs to functions derivatives
	derivatives = {"sigmoid":sigmoidDerivative}

	#sample learning rate
	learningRate = .5






#Input size is the size of the input layer
#Output size is the size of the output layer
#sizes is a vector containing the sizes of each
#of the hidden layers
def createMatrix(sizes,inputSize=None,outputSize=None):
	if inputSize == None:
	 #creates an empty vector the size of all the
	 #layers (including input and output)
		network = [0] * (len(sizes))

		for i in range(len(network)):
			network[i] = [0] * sizes[i]  

			
		
	else:
		 #creates an empty vector the size of all the
		 #layers (including input and output)
		network = [0] * (len(sizes)+2)

		network[0] = [0] * inputSize #input layer
		network[-1] = [0] * outputSize #output layer
		for i,j in zip(range(1,len(network)-1),range(len(sizes))): #skip the input and output layer
			network[i] = [0] * sizes[j]  

		#network = np.matrix(network)
	
	return network


def forwardPass(network,weights,biases,squashFunction):

	network = (np.matrix(network) + np.matrix(biases)).tolist()  #add biases to network
	
	for i in range(len(network)): #layer
		if i == len(network)-1: #when calculating outputs
			break
		for j in range(len(network[i])): #neurons in current layer
			for k in range(len(network[i+1])): #neurons in next layer
				network[i+1][k] += (network[i][j])*(weights[i][j][k]) #add the value from layer times the weight
		for m in range(len(network[i+1])):
			network[i+1][m] = MLTools.squash[squashFunction](network[i+1][m]) 

	return network

def backpropagation(network,weights,error,expectedValues,squashFunction):
	# dError_dOut = 0
	# dOut_dNet = 0
	# dNet_dW = 0
	for i in range(len(network)-1,-1,-1): #layer
		for j in range(len(network[i])): #current node
			if i == len(network)-1:
				dError_dOut = -(expectedValues[j] - network[i][j])
				dOut_dNet = MLTools.derivatives[squashFunction](network[i][j])
				dNet_dW = network[i-1][j]
				gradient = (dError_dOut*dOut_dNet*dNet_dW)
				for k in range(len(network[i-1])): #prev layer
					#i-1 = prev layer, k = which node in that layer, j = weight pointing to our current node
					weights[i-1][k][j] = weights[i-1][k][j] - (MLTools.learningRate * gradient)
					print(weights)
			#for k in range(len(network[j])-1,-1,-1): 


				
							






networkSizes = [2,2,2] #The vector containing the sizes of the layers  
network = createMatrix(networkSizes)
network[0] = [.05,.10] #initialize the input layer

weights = []
weights.append([[.15,.25],[.20,.30]])
weights.append([[.40,.50],[.45,.55]])
weights.append([[1],[1]])#no weights
#weights = np.matrix(weights)

biases = []
biases.append([0,0])
biases.append([.35,.35])
biases.append([.60,.60])
#biases = np.matrix(biases)

#print(network,"\n\n",weights,"\n\n",biases)

squashFunction = "sigmoid"  #a string with the name of the squash function
network = forwardPass(network,weights,biases,squashFunction)
print(network)
error = MLTools.squaredError(network[-1],[.01,.99]) 
print(weights)
backpropagation(network,weights,error,[.01,.99],squashFunction)








