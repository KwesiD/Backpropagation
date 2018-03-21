import operator 
import math


#sigmoid squash function
def squash(value):
		exp = math.exp(-value)
		return 1/(1+exp)


def squaredError(target,actual):
	errors = []
	total = 0
	for targetValue,actualValue in zip(target,actual):
		error = ((.5)*((targetValue-actualValue)**2))
		total += error
		errors.append(error)

	return 	errors,total




class Neuron:
	input = None
	weights = None
	bias = 0  #initialize bias to 0 for input layers.

	def setWeights(self,weights):
		self.weights = weights

	def setInput(self,input):
		self.input = input

	def setBias(self,bias):
		self.bias = bias

	def getInput(self):
		return self.input

	def getWeights(self):
		return self.weights

	def getBias(self):
		return self.bias

	def getOutputs(self): #outputs for each neuron in the next layer
		outputs = []
		for weight in self.weights:
			outputs.append((weight*self.input))
		print(outputs)
		return outputs






class InputNeuron(Neuron):

	def __init__(self,input):
		self.input = input
	
class HiddenNeuron(Neuron):

	def __init__(self,bias):
		self.bias = bias




	#override from Neuron class
	# def setInput(self,inputs,weights,bias):
	# 	#iterates over both inputs and weights at the same time
	# 	self.input = 0
	# 	for input,weight in zip(inputs,weights):
	# 		self.input += (input*weight) + self.bias


			
	
class Layer:
	neurons = None

	def __init__(self,neurons):
		self.neurons = neurons	

	def calculateInputs(self,prevLayer):
		inputSums = [0]*len(self.neurons)  #the sum of the inputs for each neuron (the first value is the input for neuron 1, etc)
		biasVector = [neuron.getBias() for neuron in self.neurons] #vector containing biases
		for neuron in prevLayer:
			inputSums = list(map(operator.add,inputSums,neuron.getOutputs())) #add each vector together

		inputSums = list(map(operator.add,inputSums,biasVector)) #add the bias vector at the end

		for neuron,input in zip(self.neurons,inputSums):
			neuron.setInput(squash(input))
			try:
				neuron.setOutput()
			except AttributeError:
				print("No method setOutput exists for this object")



	def getNeurons(self):
		return self.neurons
	
	def printLayer(self):
		for i in range(len(self.neurons)):
			print(str(i) + ": " + str(self.neurons[i].getInput()) + " " + str(self.neurons[i].getWeights()) + " " + str(self.neurons[i].getBias()))

	def getOutputs(self):
		outputs = []
		for neuron in self.neurons:
			outputs.append(neuron.getOutput())
		return outputs





class OutputNeuron(Neuron):
	output = 0

	def __init__(self,bias):
		self.bias = bias

	#copies input into output (for outputlayer neurons) since no additional operations are performed within this node
	def setOutput(self):
		self.output = self.input

	def getOutput(self):
		return self.output



	
		



def main():
	#We are creating a network with 1 hidden layer. 2 neurons in each layer (in,hidden,out)
	
	#input layer
	input1 = InputNeuron(.05)
	input1.setWeights([.15,.25])
	input2 = InputNeuron(.10)
	input2.setWeights([.20,.30])
	inputLayer = Layer([input1,input2])
	inputLayer.printLayer()

	#hidden layer
	hidden1 = HiddenNeuron(.35)
	hidden1.setWeights([.40,.50])
	hidden2 = HiddenNeuron(.35)
	hidden2.setWeights([.45,.55])
	hiddenLayer = Layer([hidden1,hidden2])
	hiddenLayer.calculateInputs(inputLayer.getNeurons())
	#hiddenLayer.printLayer()

	#output layer
	output1 = OutputNeuron(.60)
	output2 = OutputNeuron(.60)
	outputLayer = Layer([output1,output2])
	outputLayer.calculateInputs(hiddenLayer.getNeurons())
	#outputLayer.printLayer()

	layers = [inputLayer,hiddenLayer,outputLayer]

	resultVector = [.01,.99]  #the results we want 
	errorVector,totalError = squaredError(resultVector,outputLayer.getOutputs())
	print(errorVector,totalError)

	for i in range(len(layers)):
		






main()



