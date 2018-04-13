import operator 
import math
import numpy as np
from copy import copy, deepcopy
import random
import matplotlib.pyplot as plt


class NeuralNetwork():

	def __init__(self,sizes,squashFunction,learningRate,weights=None,biases=None):
		self.network = self.createMatrix(sizes)
		self.squashFunction = squashFunction
		self.errorList = []
		if(weights == None):
			self.weights = self.createMatrix(sizes)
			self.randomizeWeights()
		else:
			self.weights = weights
			if(self.compareSizes(self.network,weights) == False):
				raise Error("Network matrix and first 2 dimensions of weight matrix are not of the same dimensions.")
		#self.weights = weights
		if(biases == None):
			self.biases = self.createMatrix(sizes)
			self.randomizeBiases()
		else:
			self.biases = biases
			if(self.compareSizes(self.network,biases) == False):
				raise Error("Network and bias matrices are not of the same dimensions.")

		self.MLTools.learningRate = learningRate
		# if(len(expectedValues) != len(weights[-1])):
		# 	throw Error("Number of expected outputs does not match number of output nodes")
		# self.expectedValues = expectedValues



	# def squash(value,function):
	# 	if function == "sigmoid":
	# 		return sigmoid(value)
	# 	else:
	# 		return None


	def randomizeBiases(self):
		for i in range(len(self.biases)):
			if(i == 0):
				continue
			else:
				for j in range(len(self.biases[i])):
					self.biases[i][j] = random.random()

	def randomizeWeights(self):
		for i in range(len(self.weights)):
			if(i == len(self.weights)-1):
				for j in range(len(self.weights[i])):
					self.weights[i][j] = [1]	
			else:
				for j in range(len(self.weights[i])):
					self.weights[i][j] = [0] * len(self.weights[i+1])
					for k in range(len(self.weights[i+1])):
						self.weights[i][j][k] = random.random()


	def compareSizes(self,m1,m2):
		if(len(m1) != len(m2)):
			return False
		else:
			for i in range(len(m1)):
				if(len(m1[i]) != len(m2[i])):
					return False
		return True


	def setInput(self,inputVector):
		if(self.network == None):
			raise Exception("Attempted to assign input vector to empty network")
		elif(len(self.network[0]) != len(inputVector)):
			raise IndexError("Attempted to assign vector of size " + str(len(inputVector)) 
				+ " to matrix column with size " + str(len(self.network[0]))) 
		else:
			self.network[0] = inputVector



	def flush(self):
		for i in range(len(self.network)):
			if(i == 0):
				continue
			else:
				for j in range(len(self.network[i])):
					self.network[i][j] = 0
			


	class MLTools(object):

		#sigmoid squash function
		def sigmoid(value):
			exp = math.exp(-value)
			return 1/(1+exp)

		def sigmoidDerivative(value):
			return (value)*(1-value)

		def relu(netInput):
			return max(0, netInput)

		def reluDerivative(value):
			if(value > 0):
				return 1
			else:
				return 0

		@staticmethod
		def squaredError(expected,actual):
			total = 0
			#errorVector = []
			for e,a in zip(expected,actual):
				error = (.5) * ((e-a)**2)  #Σ (1/2 (expected−actual)^2)
				total += error
				#errorVector.append(error)
			return total #,errorVector


		#Dictionary containing refs to functions
		squash = {"sigmoid":sigmoid,"relu":relu}

		#Dictionary containing refs to functions derivatives
		derivatives = {"sigmoid":sigmoidDerivative,"relu":reluDerivative}

		#initial learning rate
		#learningRate = .5






	#Input size is the size of the input layer
	#Output size is the size of the output layer
	#sizes is a vector containing the sizes of each
	#of the hidden layers
	def createMatrix(self,sizes,inputSize=None,outputSize=None):
		if inputSize == None:
		 #creates an empty vector the size of all the
		 #layers (including input and output)
			newNetwork = [0] * (len(sizes))

			#populates the vector with vectors for each layer
			for i in range(len(newNetwork)):
				newNetwork[i] = [0] * sizes[i]  

		else:
			 #creates an empty vector the size of all the
			 #layers (including input and output)
			newNetwork = [0] * (len(sizes)+2)

			newNetwork[0] = [0] * inputSize #input layer
			newNetwork[-1] = [0] * outputSize #output layer
			for i,j in zip(range(1,len(newNetwork)-1),range(len(sizes))): #skip the input and output layer
				newNetwork[i] = [0] * sizes[j]  

			#network = np.matrix(network)
		
		return newNetwork


	def addMatrices(self,m1,m2):
		for i in range(len(m1)):
			for j in range(len(m1[i])):
				m1[i][j] += m2[i][j]
		return m1


	def feedForward(self,inputs):

		self.setInput(inputs) #sets the inputs
		self.flush() #flush the previous data in the network feed
		#self.network = (np.matrix(self.network) + np.matrix(self.biases)).tolist()  #add biases to network
		self.network = self.addMatrices(self.network,self.biases)
		
		for i in range(len(self.network)): #layer
			if i == len(self.network)-1: #when calculating outputs
				break
			for j in range(len(self.network[i])): #neurons in current layer
				for k in range(len(self.network[i+1])): #neurons in next layer
					self.network[i+1][k] += (self.network[i][j])*(self.weights[i][j][k]) #add the value from layer times the weight
			for m in range(len(self.network[i+1])):
				self.network[i+1][m] = self.MLTools.squash[self.squashFunction](self.network[i+1][m])
			

		return self.network[-1] #output


	def backpropagation(self,expectedValues):
		error = self.MLTools.squaredError(self.network[-1],expectedValues)
		#errorVector = [] #Vector containing the dError/dOut values 
		#outVector = [] #Vector containing the dOut/dNet values
		originalWeights = deepcopy(self.weights) #deepcopy prevents the references from being copied rather than values

		for i in range(len(self.network)-1,-1,-1): #layer
			for j in range(len(self.network[i])): #current node
				if i == len(self.network)-1: #for the output layer
					dError_dOut = -(expectedValues[j] - self.network[i][j])
					dOut_dNet = self.MLTools.derivatives[self.squashFunction](self.network[i][j])
					#errorVector.append(dError_dOut)
					#outVector.append(dOut_dNet)
					for k in range(len(self.network[i-1])):
						dNet_dW = self.network[i-1][k]
						gradient = (dError_dOut*dOut_dNet*dNet_dW)
						
						#print(str(gradient))
						
						# 	#i-1 = prev layer, k = which node in that layer, j = weight pointing to our current node
						self.weights[i-1][k][j] = originalWeights[i-1][k][j] - (self.MLTools.learningRate * gradient)
						
						#print(weights[i-1][k][j])

				elif i == 0:
					break
				else:
					dError_dOut = 0
					for l in range(len(self.network[i+1])): #the layer ahead, analogous to j
						#these derivatives are relative to the next layer
						dE_dNet = -(expectedValues[l] - self.network[i+1][l])
						dOut_dNet = self.MLTools.derivatives[self.squashFunction](self.network[i+1][l])
						dEl_doutk = dE_dNet * dOut_dNet * originalWeights[i][j][l]     #the error of the next node/output of this node
						dError_dOut += dEl_doutk
					#these derivatives are for the current layer
					dOut_dNet = self.MLTools.derivatives[self.squashFunction](self.network[i][j])
					for k in range(len(self.network[i-1])): #the previous layer
						dNet_dW = self.network[i-1][k]
						gradient = (dError_dOut * dOut_dNet * dNet_dW)
						self.weights[i-1][k][j] = originalWeights[i-1][k][j] - (self.MLTools.learningRate * gradient)
						
						#print(weights[i-1][k][j])

		return error

	def train(self,inputs,expected,showErrors=False,showOutput=False):
		#print(self.weights)
		output = self.feedForward(inputs)
		#print("output",output)
		errors = self.backpropagation(expected)
		self.errorList.append(errors)
		if(showErrors and not showOutput):
			return errors
		elif(showOutput and not showErrors):
			return output
		else:
			return errors,output

def createAdd(lower,upper,size):
	inputs = []
	outputs = []
	for i in range(size):
		n1 = random.randint(lower,upper)
		n2 = random.randint(lower,upper)
		inputs.append([lower+n1,upper-n2])
		outputs.append((lower+n1)+(upper-n2))
	return inputs,outputs

# networkSizes = [2,2,2] #The vector containing the sizes of the layers  
# #network = createMatrix(networkSizes)
# inputs = [.05,.10] #initialize the input layer

# weights = []
# weights.append([[.15,.25],[.20,.30]])
# weights.append([[.40,.50],[.45,.55]])
# weights.append([[1],[1]])#no weights
# #weights = np.matrix(weights)

# biases = []
# biases.append([0,0])
# biases.append([.35,.35])
# biases.append([.60,.60])
# #biases = np.matrix(biases)

# targets = [.01,.99]
# squashFunction = "sigmoid"  #a string with the name of the squash function
# learningRate = .5
# #Initialize Netword
# network = NeuralNetwork(networkSizes,squashFunction,learningRate,weights=weights,biases=biases)
# # network.train(inputs,targets)
# # print(network.network)
# for i in range(10000): #num epochs
# 	network.train(inputs,targets)
# 	#print(network.weights,"\n","-"*10)
# print(network.feedForward(inputs),"\n","-"*10)



#XOR
networkSizes = [2,2,1] #The vector containing the sizes of the layers
inputs = [[0,0],[0,1],[1,0],[1,1]]
targets = [0,1,1,0]
squashFunction = "sigmoid"
learningRate = .1
network = NeuralNetwork(networkSizes,squashFunction,learningRate)
for i in range(50000): #num epochs
	for j in range(len(inputs)):
		network.train(inputs[j],[targets[j]])
for i in range(len(inputs)):
	print(network.feedForward(inputs[i]))
print(network.weights)

#plt.plot(network.errorList)
#plt.show()


# #addition: doesnt work
# networkSizes = [2,2,1] #The vector containing the sizes of the layers
# inputs,targets = createAdd(0,50,10)
# print(targets)
# validation_inputs,validation_targets = createAdd(50,100,10)
# squashFunction = "relu"
# learningRate = 1
# network = NeuralNetwork(networkSizes,squashFunction,learningRate)
# for i in range(100000): #num epochs
# 	for j in range(len(inputs)):
# 		network.train(inputs[j],[targets[j]])
# for i in range(len(validation_inputs)):
# 	print(network.feedForward(validation_inputs[i])," | ",validation_targets[i])


