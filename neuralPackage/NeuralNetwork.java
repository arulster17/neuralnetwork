package neuralPackage;
public class NeuralNetwork {
    public Layer[] layers;
    
    // Neural Network Constructor
    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        for(int i = 0; i < layerSizes.length - 1; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }
    }
    
    public String printArray(double[] arr) {
        String finalStr = "";
        for(double el : arr) {
            finalStr += el + " ";
        }
        return finalStr;
    }
    
    // Computes outputs of network for given inputs
    public double[] computeOutputs(double[] inputs) {
        for(Layer layer : layers) {
            inputs = layer.computeOutputs(inputs);
        }
        return inputs;
    }
    
    // Computes outputs of network and calculates which output node has the highest value
    public int classify(double[] inputs) {
        double[] outputs = softMax(computeOutputs(inputs));
        int bestIndex = 0;
        double bestScore = outputs[0];
        for(int i = 1; i < outputs.length; i++) {
            if(outputs[i] > bestScore) {
                bestScore = outputs[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
    
    // One iteration of Gradient Descent
    public void learn(DataPoint[] trainingData, double learnRate) {
        // Backpropagation algorithm (adds all the gradients for every data point)
        for(DataPoint D : trainingData) {
            updateAllGradients(D);
        }
        
        // Update all of the weights and biases
        applyAllGradients(learnRate / trainingData.length);
        
        // Reset the gradients
        clearAllGradients();
        
    }
    
    // Backpropagation
    void updateAllGradients(DataPoint dataPoint) {
        // Run inputs and store needed data in layers
        computeOutputs(dataPoint.inputs);
        
        // Update gradients of output layer
        Layer outputLayer = layers[layers.length - 1];
        double[] nodeValues = outputLayer.calculateOutputNodeValues(dataPoint.expectedOutputs);
        outputLayer.updateGradients(nodeValues);
        
        // Update gradients of hidden layers by looping backwards
        for(int hiddenLayerIndex = layers.length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex --) {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.calculateHiddenNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.updateGradients(nodeValues);
        }
        
    }
    
    // Apply the gradients to each layer to update the weights and biases
    void applyAllGradients(double learnRate) {
        for(Layer layer : layers) {
            layer.applyGradient(learnRate);
        }
    }
    
    // Reset gradients to 0 for next training batch
    void clearAllGradients() {
        for(Layer layer : layers) {
            layer.costGradientW = new double[layer.numInputNodes][layer.numOutputNodes];
            layer.costGradientB = new double[layer.numOutputNodes];
        }
    }
    
    // Loss function
    double cost(DataPoint dataPoint) {
        double[] outputs = computeOutputs(dataPoint.inputs);
        Layer outputLayer = layers[layers.length - 1];
        double cost = 0;
        
        for(int outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
            cost += outputLayer.nodeCost(outputs[outputIndex], dataPoint.expectedOutputs[outputIndex]);
        }
        
        return cost;
    }
    
    // Average loss function
    double cost(DataPoint[] datapoints) {
        double totalCost = 0;
        for(DataPoint D : datapoints) {
            totalCost += cost(D);
        }
        return (totalCost / datapoints.length);
    }
    
    double[] softMax(double[] input) {
        double expSum = 0;
        for(double num : input) {
            expSum += Math.exp(num);
        }
        double[] result = new double[input.length];
        for(int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i]) / expSum;
        }
        return result;
    }
}
