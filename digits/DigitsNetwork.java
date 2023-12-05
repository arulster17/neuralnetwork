package digits;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import neuralPackage.*;

import digits.NoiseStuff;

public class DigitsNetwork {
    
    public static String strArray(double[] arr) {
        String finalStr = "";
        for(double el : arr) {
            finalStr += el + " ";
        }
        return finalStr;
    }
    
    public static double maxVal(double[][] arr) {
        double max = arr[0][0];
        for(double[] row : arr) {
            for(double el : row) {
               if (el<max) max = el; 
            }
        }
        return max;
    }
    
    public static DataPoint[] shuffle(DataPoint[] arr) {
        Random r = new Random();
        for(int i = arr.length; i > 1; i--) {
            swap(arr, i-1, r.nextInt(i));
        }
        return arr;
    }
    public static void swap(DataPoint[] arr, int i, int j) {
        DataPoint temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    public static void main(String [] args) throws IOException {
        int[] layers = {784, 500, 200, 50, 10};
        FileReader fr = new FileReader("digits/readableTrainingData/trainingAnswers"); 
        BufferedReader br = new BufferedReader(fr);
        String[] answers = br.readLine().split(" ");
        
        
        
        double[][] inputs = new double[120000][784];
        String[] ss;
        
        int k=0;
        for(int i = 1; i < 61; i++) {
            fr = new FileReader("digits/readableTrainingData/trainingData" + i); 
            br = new BufferedReader(fr);
            while(br.ready()) {
                ss = br.readLine().split(" ");
                for(int z = 0; z < 784; z++) {
                    inputs[2*k][z] = Double.valueOf(ss[z]) / 256;
                    inputs[2*k+1][z] = Double.valueOf(ss[z]) / 256;
                }
                k++;
                
            }
        }
        /*
        Random r = new Random();
        int s = r.nextInt(60000);
        NoiseStuff.printImage(inputs[s]);
        for(int i = 0; i < 10; i++) {
            System.out.println(); 
        }
        NoiseStuff.printImage(NoiseStuff.randomize(inputs[s]));
        
        System.out.println(answers[s]);
        */
        br.close();
        double[] trainingAnswer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        
        DataPoint[] dataset = new DataPoint[120000];
        
        for(int i = 0; i < 120000; i++) {
            trainingAnswer[Integer.valueOf(answers[i/2])] = 1;
            dataset[i] = new DataPoint(NoiseStuff.randomize(inputs[i]), trainingAnswer.clone());
            trainingAnswer[Integer.valueOf(answers[i/2])] = 0;
        }
        long start = System.currentTimeMillis();
        for(int t = 0; t < 1; t++) {
            NeuralNetwork digitNet = new NeuralNetwork(layers);
            
            int batchSize = 30;
    
            
            //System.out.println("starting training");
            for(int i = 0; i < dataset.length / batchSize; i++) {
                digitNet.learn(Arrays.copyOfRange(dataset, batchSize*i, batchSize*(i+1)), 1.25);
                System.out.println("Batch " + (i+1) + " of " + (dataset.length / batchSize) + " done.");
            }
            //System.out.println("Training Completed");
            //System.out.println("Time Elapsed: " + (System.currentTimeMillis() - start) + "ms");
    
            
            
            fr = new FileReader("digits/readableTestingData/testingData"); 
            br = new BufferedReader(fr);
            double[][] testInputs = new double[10000][784];
            k = 0;
            while(br.ready()) {
                ss = br.readLine().split(" ");
                for(int z = 0; z < 784; z++) {
                    // values between 0 and 1 so we have more manageable numbers
                    testInputs[k][z] = Double.valueOf(ss[z]) / 256;
                }
                k++;
            }
            
            FileReader fr2 = new FileReader("digits/readableTestingData/testingAnswers"); 
            br = new BufferedReader(fr2);
            
            String[] testAnswers = br.readLine().split(" ");
            double numCorrect = 0.0;
            for(int i = 0; i < 10000; i++) {
                
                int n = digitNet.classify(testInputs[i]);
                if(Integer.valueOf(testAnswers[i]) == n) {
                    numCorrect++;
                }
            }
            //for(Layer layer : digitNet.layers) {
            //    System.out.println(maxVal(layer.weights));
            //}
            System.out.println("Percentage Correct in trial " + t + ": " + (100*numCorrect/10000) + "%");
            System.out.println("Time Elapsed: " + (System.currentTimeMillis() - start) + "ms");
            fr.close();
            
            
            
        }
    }
}
