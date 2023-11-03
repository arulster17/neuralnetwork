import java.util.Random;

public class Layer {
    int numInputNodes;
    int numOutputNodes;
    
    double[] inputs;
    
    double[][] weights;
    double[] biases;
    
    double[][] costGradientW;
    double[] costGradientB;
    
    double[] activations;
    double[] weightedInputs;
    
    
    // Layer Constructor
    public Layer(int numInputNodes, int numOutputNodes) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        inputs = new double[numInputNodes];
        weights = new double[numInputNodes][numOutputNodes];
        biases = new double[numOutputNodes];
        activations = new double[numOutputNodes];
        weightedInputs = new double[numOutputNodes];
        costGradientW = new double[numInputNodes][numOutputNodes];
        costGradientB = new double[numOutputNodes];
    }
    
    // Update weights and biases based on gradients and learn rate
    public void applyGradient(double learnRate) {
        for(int outIndex = 0; outIndex < numOutputNodes; outIndex++) {
            biases[outIndex] -= costGradientB[outIndex] * learnRate;
            for(int inIndex = 0; inIndex < numInputNodes; inIndex++) {
                weights[inIndex][outIndex] -= costGradientW[inIndex][outIndex] * learnRate;
            }
        }
    }
    
    // Compute output given inputs
    double[] computeOutputs(double[] inputArray) {
        inputs = inputArray;
        double output;
        for(int outputIndex = 0; outputIndex < numOutputNodes; outputIndex++) {
            output = biases[outputIndex];
            for(int inputIndex = 0; inputIndex < numInputNodes; inputIndex++) {
                output += weights[inputIndex][outputIndex]*inputArray[inputIndex];
            }
            weightedInputs[outputIndex] = output;
            activations[outputIndex] = activationFunction(output);
        }
        return activations;
    }
    
    
    // Calculate "node values" for output layer
    // hopefully this works, I did something different without the NetworkLearnData stuff :(
    double[] calculateOutputNodeValues(double[] expectedOutputs) {
        
        double[] nodeValues = new double[expectedOutputs.length];
        double costDerivative, activationDerivative;
        for(int i = 0; i < nodeValues.length; i++) {
            // cost/activation * activation/weightedInput
            costDerivative = nodeCostDerivative(activations[i], expectedOutputs[i]);
            activationDerivative = activationFunctionDerivative(weightedInputs[i]);
            nodeValues[i] = costDerivative * activationDerivative;
            
        }
        
        return nodeValues;
    }
    
    double[] calculateHiddenNodeValues(Layer oldLayer, double[] oldNodeValues) {
        double[] newNodeValues = new double[numOutputNodes];
        double newNodeValue;
        for(int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            newNodeValue = 0;
            for(int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                // weightedInput/input
                newNodeValue += (oldLayer.weights[newNodeIndex][oldNodeIndex] * oldNodeValues[oldNodeIndex]);
            }
            newNodeValue *= activationFunctionDerivative(weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }
        
        return newNodeValues;
    }
    
    
    // Update gradients using node values
    void updateGradients(double[] nodeValues) {
        for(int outIndex = 0; outIndex < numOutputNodes; outIndex++) {
            for(int inIndex = 0; inIndex < numInputNodes; inIndex++) {
                // derivative of cost WRT weight
                costGradientW[inIndex][outIndex] += (inputs[inIndex] * nodeValues[outIndex]);
            }
            // derivative of cost WRT bias
            costGradientB[outIndex] += 1 * nodeValues[outIndex];
        }
    }
    
    
    // Activation Function (Sigmoid). Squishes output between 0 and 1
    double activationFunction(double weightedInput) {
        return 1.0 / (1 + Math.exp(-weightedInput));
    }
    
    // Partial Derivative of activationFunction w.r.t. weightedInput
    double activationFunctionDerivative(double weightedInput) {
        double activation = activationFunction(weightedInput);
        return activation * (1-activation);
    }
    
    // Cost of each node
    double nodeCost(double outputActivation, double desiredOutput) {
        return (outputActivation - desiredOutput)*(outputActivation - desiredOutput);
    }
    
    // Partial Derivative of nodeCost w.r.t. outputActivation
    double nodeCostDerivative(double outputActivation, double desiredOutput) {
        return 2*(outputActivation - desiredOutput);
    }
    
    
    // Initialize Random Weights (using Sebastian's weight generation)
    void initializeRandomWeights() {
        Random R = new Random();
        for(int outputIndex = 0; outputIndex < numOutputNodes; outputIndex++) {
            for(int inputIndex = 0; inputIndex < numInputNodes; inputIndex++) {
                weights[inputIndex][outputIndex] = (R.nextDouble() * 2 - 1) / Math.sqrt(numInputNodes);
            }
        }
    }
    
    
    
}