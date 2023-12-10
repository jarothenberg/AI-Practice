import math
import random

import drawingReader
import gz


class Neuron:
    def __init__(self, weights, activation, bias):
        self.weights = weights
        self.activation = activation
        self.bias = bias


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons


class AI:
    def __init__(self, layers):
        self.layers = layers

    def process(self, inData):
        layers = self.layers
        for k in range(len(layers)):
            curLayer = layers[k]
            curNeurons = curLayer.neurons
            if k == 0:
                for i in range(len(curNeurons)):
                    curNeurons[i].activation = inData[i]
                # normalizeActivations(layers[k])
            elif k == len(layers) - 1:
                for i in range(len(curNeurons)):
                    curNeurons[i].activation = getActivation(layers[k - 1], curNeurons[i])
                # normalizeActivations(layers[k])
                return getResults(layers[k]), finalGuess(layers[k])
            else:
                for i in range(len(curNeurons)):
                    curNeurons[i].activation = getActivation(layers[k - 1], curNeurons[i])
                # normalizeActivations(layers[k])

    def assess(self, inData):
        totalCorrect = 0
        numTests = len(inData)
        for i in range(numTests):
            results = self.process(inData[i][0])
            if results[1] == inData[i][1][0]:
                totalCorrect += 1
                print("Assessing Test " + str(i + 1) + ": CORRECT")
            else:
                print("Assessing Test " + str(i + 1) + ": INCORRECT")
            # print("Guess: " + str(results[1]) + " True: " + str(inData[i][1][0]))
        return totalCorrect / numTests

    def train(self, trainingData):  # works with Tuple
        outLayerIndex = len(self.layers) - 1
        sumCost = 0
        for i in range(len(trainingData)):
            [result, guess] = self.process(trainingData[i][0])
            answer = (trainingData[i][1].index((max(trainingData[i][1]))))
            print("trueOut: " + str(trainingData[i][1]) + "\tanswer: " + str(answer) + "\toutLayer: " + str(
                [round(item, 2) for item in result]) + "\tguess: " + str(guess))
            curCost = cost(self.layers[outLayerIndex], trainingData[i][1])
            sumCost += curCost
            curAveCost = sumCost / (i + 1)
            print("dataNum(" + str(i) + ")\tCost: " + str(curCost) + "\taveCost: " + str(curAveCost))
            # tweakValues(self, trainingData[i][1])
            # TODO have learning rate be a function of cost (look into https://www.jeremyjordan.me/nn-learning-rate/)
            # learningRate = curCost ** 3 / 5
            learningRate = 0.5
            # TODO ^
            tweakValues2(self, trainingData[i][1], learningRate)
        return sumCost / len(trainingData)

    def train2(self, trainingData, numBatches):
        outLayerIndex = len(self.layers) - 1
        aveCost = 0
        batchedData = dataSplit(trainingData, numBatches)
        for i in range(len(batchedData)):
            for j in range(len(batchedData[i])):
                self.process(batchedData[i][j][0])
                aveCost += cost(self.layers[outLayerIndex], batchedData[i][j][1])
        return aveCost / len(trainingData)


class Data:
    def __init__(self, test, val):
        self.test = test
        self.val = val


def sigmoid(x):
    # print("SIGMOID x; x = " + str(x))
    if x < 0:
        return 1 - 1 / (1 + math.e ** x)
    else:
        return 1 / (1 + math.e ** (-1 * x))


def zeroCenteredSigmoid(x):
    return sigmoid(x) - 0.5


def sigmoidDerivative(x):
    # print("SIGMOIDDERIVATIVE x; x = " + str(x))
    return sigmoid(x) * (1 - sigmoid(x))


def rectifiedLinearUnit(x):
    if x <= 0:
        return 0.000001 * x
    return x


def reLUDerivative(x):
    if x <= 0:
        return 0.000001
    return 1


def displayData(data):
    if len(data[1]) == 1:
        answer = data[1][0]
    else:
        answer = data[1].index(max(data[1]))
    print(answer)
    currentData = data[0]
    count = 0
    for i in range(28):
        line = ""
        for j in range(28):
            val = currentData[count]
            count += 1
            if val > 0:
                line += "X"
            else:
                line += " "
        print(line)


def finalGuess(layer):
    maxNeuron = 0
    neurons = layer.neurons
    for i in range(len(neurons)):
        neuron = neurons[i]
        if neuron.activation > neurons[maxNeuron].activation:
            maxNeuron = i
    return maxNeuron


def getResults(layer):
    neurons = layer.neurons
    results = [0] * len(neurons)
    for i in range(len(neurons)):
        results[i] = neurons[i].activation
    return results


def normalizeActivations(layer):
    minActivation = 9999999999999999
    for i in range(len(layer.neurons)):
        neuron = layer.neurons[i]
        if neuron.activation < minActivation:
            minActivation = neuron.activation
    for i in range(len(layer.neurons)):
        neuron = layer.neurons[i]
        neuron.activation -= minActivation

    maxActivation = 0.00000000001
    for i in range(len(layer.neurons)):
        neuron = layer.neurons[i]
        if neuron.activation > maxActivation:
            maxActivation = neuron.activation
    for i in range(len(layer.neurons)):
        neuron = layer.neurons[i]
        neuron.activation = neuron.activation / maxActivation


def dataSplit(trainingData, numBatches):
    length = len(trainingData)
    splitData = [0] * numBatches
    count = 0
    remainder = length % numBatches
    for i in range(numBatches):
        currentBatch = [0] * (length // numBatches)
        for j in range(len(currentBatch)):
            currentBatch[j] = trainingData[count]
            count += 1
        splitData[i] = currentBatch
    for i in range(remainder):
        splitData[numBatches - 1 - i].append(trainingData[count])
        count += 1
    return splitData


def createAI(layerSizes):
    layersNum = len(layerSizes)
    layersArray = [0] * layersNum

    for k in reversed(range(layersNum)):
        if k == 0:
            currentLayerSize = layerSizes[k]
            preLayerSize = 1
            weightsArray = [[0 for x in range(preLayerSize)] for y in range(currentLayerSize)]
            bias = [0] * currentLayerSize
            neuronsArray = [0] * currentLayerSize
            neuronActivationArray = [0] * currentLayerSize  # set it to starting input activation
            for i in range(currentLayerSize):
                for j in range(preLayerSize):
                    weightsArray[i][j] = 1  # to be changed
                neuronsArray[i] = Neuron(weightsArray[i], neuronActivationArray[i], bias[i])
            layersArray[k] = Layer(neuronsArray)
        elif k == layersNum - 1:
            currentLayerSize = layerSizes[k]
            preLayerSize = layerSizes[k - 1]  # necessary as no next layer, just single output
            weightsArray = [[0 for x in range(preLayerSize)] for y in range(currentLayerSize)]
            bias = [0] * currentLayerSize
            neuronsArray = [0] * currentLayerSize
            neuronActivationArray = [0] * currentLayerSize
            for i in range(currentLayerSize):
                bias[i] = random.uniform(-1, 1)
                for j in range(preLayerSize):
                    weightsArray[i][j] = random.uniform(-1, 1)  # to be changed
                neuronsArray[i] = Neuron(weightsArray[i], neuronActivationArray[i], bias[i])
            layersArray[k] = Layer(neuronsArray)
        else:
            currentLayerSize = layerSizes[k]
            preLayerSize = layerSizes[k - 1]
            weightsArray = [[0 for x in range(preLayerSize)] for y in range(currentLayerSize)]
            bias = [0] * currentLayerSize
            neuronsArray = [0] * currentLayerSize
            neuronActivationArray = [0] * currentLayerSize
            for i in range(currentLayerSize):
                bias[i] = random.uniform(-1, 1)
                for j in range(preLayerSize):
                    weightsArray[i][j] = random.uniform(-1, 1)  # to be changed
                neuronsArray[i] = Neuron(weightsArray[i], neuronActivationArray[i], bias[i])
            layersArray[k] = Layer(neuronsArray)
    return AI(layersArray)


def getActivation(preLayer, neuron):
    activation = neuron.bias
    preNeurons = preLayer.neurons
    for i in range(len(preNeurons)):
        activation += neuron.weights[i] * preNeurons[i].activation
    # return rectifiedLinearUnit(activation)
    return sigmoid(activation)
    # return zeroCenteredSigmoid(activation)


def cost(layerOut, targetVals):
    cost = 0
    neurons = layerOut.neurons
    for i in range(len(neurons)):
        cost += (neurons[i].activation - targetVals[i]) ** 2
    return cost


def tweakValues(ai, targetVals):  # targetVal is expected outputs of ai
    learningRate = 0.1
    for i in reversed(range(1, len(ai.layers))):
        layer = ai.layers[i]
        preLayer = ai.layers[i - 1]
        preTargetVals = [0] * len(preLayer.neurons)
        for j in range(len(layer.neurons)):
            neuron = layer.neurons[j]
            deltaBias = 0
            for k in range(len(neuron.weights)):
                preNeuron = preLayer.neurons[k]
                z = neuron.weights[k] * preNeuron.activation + neuron.bias
                part1 = preNeuron.activation
                part2 = sigmoidDerivative(z)
                # part2 = reLUDerivative(z)
                part3 = 2 * (neuron.activation - targetVals[j])
                deltaWeight = part1 * part2 * part3
                neuron.weights[k] -= deltaWeight * learningRate
                deltaBias += part2 * part3
                preTargetVals[k] += neuron.weights[k] * part2 * part3
            deltaBias /= len(preLayer.neurons)
            neuron.bias -= deltaBias * learningRate
            # print(neuron.weights)
        targetVals = [0] * len(preLayer.neurons)
        for j in range(len(preLayer.neurons)):
            targetVals[j] = preTargetVals[j] + preLayer.neurons[j].activation


def tweakValues2(ai, targetVals, learningRate):  # targetVal is expected outputs of ai
    mixedList = []
    part3 = 0
    for i in reversed(range(1, len(ai.layers))):
        layer = ai.layers[i]
        preLayer = ai.layers[i - 1]
        neuronsActivationList = []
        for j in range(len(layer.neurons)):
            neuron = layer.neurons[j]
            neuronsActivationList.append(neuron.activation)
            for k in range(len(neuron.weights)):
                preNeuron = preLayer.neurons[k]
                part1 = preNeuron.activation
                part2 = sigmoidDerivative(neuron.activation)
                # part2 = reLUDerivative(neuron.activation)
                if i == len(ai.layers) - 1:
                    part3 = 2 * (neuron.activation - targetVals[j])
                else:
                    part3 = 0
                    numSkip = len(layer.neurons)
                    nextLayer = ai.layers[i + 1]
                    # print("TESTCONNECT" + str(j) + str(k) + str(i))
                    # print(mixedList)
                    for n in range(len(nextLayer.neurons)):
                        part3 += mixedList[j + n * numSkip][0] * mixedList[j + n * numSkip][1] * \
                                 mixedList[j + n * numSkip][2]
                mixedList.append([neuron.weights[k], part2, part3])
                neuron.weights[k] -= learningRate * part1 * part2 * part3
            neuron.bias -= learningRate * part2 * part3


# columnsNum = 28
# rowsNum = 28
# inputArray = [1] * rowsNum * columnsNum
# trainingData = [[inputArray, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
#                 [inputArray, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
#                 [inputArray, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]]

# layerSizes = [28 * 28, 16, 16, 10]
# ai = createAI(layerSizes)
# ai.process(inputArray)
# print(ai.layers[3].neurons[0].activation)
# print(ai.train(trainingData))

# print(ai.train2(trainingData, 2))


trainingData = gz.load_data_wrapper()[0]
layerSizes = [28 * 28, 16, 16, 10]
ai = createAI(layerSizes)
print(ai.train(trainingData))

# TODO get validation data working
testData = gz.load_data_wrapper()[1]
print(ai.assess(testData))

done = True
while not done:
    data = drawingReader.getData()
    print(ai.process(data))

#
# layerSizes = [2, 10, 10, 10, 2]
# ai = createAI(layerSizes)
# print(ai.layers[0].neurons[0].weights[0])
# ai.layers[1].neurons[0].weights[0] = 0.3
# ai.layers[1].neurons[0].weights[1] = -0.4
# ai.layers[1].neurons[1].weights[0] = 0.2
# ai.layers[1].neurons[1].weights[1] = 0.6
# ai.layers[2].neurons[0].weights[0] = 0.7
# ai.layers[2].neurons[0].weights[1] = 0.5
# ai.layers[2].neurons[1].weights[0] = -0.3
# ai.layers[2].neurons[1].weights[1] = -0.1
# ai.layers[1].neurons[0].bias = 0.25
# ai.layers[1].neurons[1].bias = 0.45
# ai.layers[2].neurons[0].bias = 0.15
# ai.layers[2].neurons[1].bias = 0.35
#
#
#
# print(ai.process(inputArray)[0])
# for i in range(1000):
#     A = random.uniform(-1, 1)
#     B = random.uniform(-1, 1)
#     inputArray = [A, B]
#     if A >= B:
#         trainingData = [[inputArray, [1, 0]]]
#     else:
#         trainingData = [[inputArray, [0, 1]]]
#     (ai.train(trainingData))
#     print("A: " + str(A) + " B: " + str(B))
# print(ai.process(inputArray)[0])


# for layer in ai.layers:
#     for neuron in layer.neurons:
#         print("weights:")
#         for weight in neuron.weights:
#             print(weight)
#         print("bias: " + str(neuron.bias))


# done = False
# while (not done):
#     userInA = input("Please enter the first number to be compared or DONE to cancel: ")
#     userInB = input("Please enter the second number to be compared or DONE to cancel: ")
#     if str.upper(userInA) == "DONE" or str.upper(userInB) == "DONE":
#         done = True
#     else:
#         userInA = float(userInA)
#         userInB = float(userInB)
#         print(ai.process([userInA, userInB])[0])


# print(trainingData[0][0][0])


# displayData(trainingData[7:10])


# print(data)

# print(trainingData3[0][1])
