package digits;

import java.util.Random;

public class NoiseStuff {
    
    // assumes 28x28 = 784 length array
    static void printImage(double[] input) {
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                if(input[i*28+j] < 0.05) {
                //if(false) {
                    //System.out.print("    ");
                    System.out.print(". ");
                }
                else {
                    System.out.print(String.format("%.1f", input[i*28+j]-0.01).charAt(2) + " ");
                }
                
            }
            System.out.println();
        }
    }
    
    static final double NOISE_PROBABILITY = .1;
    static final double NOISE_STRENGTH = .1; // NOISE_STRENGTH k represents potential a -k/2 to k/2 shift
    static final double MAX_ROTATION_ANGLE = Math.PI / 6;
    static final double MAX_SCALE = 1.25;
    static final double MIN_SCALE = .8;
    static final double MAX_DISPLACEMENT = 5;

    // assumes the input is a standardized image represented by a double array of length 784
    public static double[] randomize(double[] input) {
        return addNoise(transpose(rotate(scale(input))));
    }
    
    public static double[] addNoise(double[] inputs) {
        Random rand = new Random();
        double[] result = new double[784];
        double noise;
        for(int i = 0; i < 784; i++) {
            if(rand.nextDouble() < NOISE_PROBABILITY) {
                noise = 2*(rand.nextDouble() - .5) * NOISE_STRENGTH;
                result[i] = Math.min(Math.max(inputs[i] + noise, 0), 1);
                //System.out.println("Replaced " + inputs[i] + " with " + result[i]);
                // x < 0 -> 0, 0 < x < 1 -> x, x > 1 -> 1
            }
            else {
                result[i] = inputs[i];
            }
        }
        System.out.println("Added Noise");
        return result;
    }
    
    public static double estimateValue(double x, double y, double[] input) {
        
        int a = (int) Math.floor(x);
        int b = (int) Math.floor(y);
        double ra = x - a;
        double rb = y - b;
        
        
        
        // f(7.6,10.9)= f(7, 10.9)+(0.6)*(f(8, 10.9)-f(7, 10.9))
        // f(7, 10.9) = f(7, 10) + (0.9)*(f(7, 11)-f(7, 10))
        // f(8, 10.9) = f(8, 10) + (0.9)*(f(8, 11)-f(8, 10))
        
        // f(7.6,10.9)= (f(7, 10) + (0.9)*(f(7, 11)-f(7, 10)))
        //              +(0.6)*(
        //                      (f(8, 10) + (0.9)*(f(8, 11)-f(8, 10)))
        //                      -(f(7, 10) + (0.9)*(f(7, 11)-f(7, 10))))
        
        // f(a+ra,b+rb)= (f(a, b) + (rb)*(f(a, b+1)-f(a, b)))
        //              +(ra)*(
        //                      (f(a+1, b) + (rb)*(f(a+1, b+1)-f(a+1, b)))
        //                      -(f(a, b) + (rb)*(f(a, b+1)-f(a, b))))
        
        return (getVal(a, b, input) + rb*(getVal(a, b+1, input)-getVal(a, b, input)))
               + ra * (
                 (getVal(a+1, b, input) + rb * (getVal(a+1, b+1, input)-getVal(a+1, b, input)))
                  -(getVal(a, b, input) + rb * (getVal(a, b+1, input)-getVal(a, b, input)))
                      );
        
    }
    
    public static double getVal(int x, int y, double[] input) {
        if(x < 0 || x >= 28 || y < 0 || y >= 28) {
            return 0;
        }
        return input[x*28 + y];
    }
    
    public static double[] rotate(double[] input) {
        // max rotate of -ø to ø
        Random rand = new Random();
        double[] result = new double[784];
        double rtangle;
        rtangle = MAX_ROTATION_ANGLE*2*(rand.nextDouble()-0.5);
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                // translate 0->27 -> -13.5->13.5
                double x = j - 13.5;
                double y = i - 13.5;
                double rotX = x*Math.cos(rtangle)-y*Math.sin(rtangle);
                double rotY = x*Math.sin(rtangle)+y*Math.cos(rtangle);
                //compute place in array
                double rotJ = rotX + 13.5;
                double rotI = rotY + 13.5;
                // say you want to estimate f(7.6, 10.9)
                // f(7.6,10.9)= f(7, 10.9)+(0.6)*(f(8, 10.9)-f(7, 10.9))
                // f(7, 10.9) = f(7, 10) + (0.9)*(f(7, 11)-f(7, 10))
                // f(8, 10.9) = f(8, 10) + (0.9)*(f(8, 11)-f(8, 10))
                result[28*i + j] = estimateValue(rotI, rotJ, input);
            }
        }
        System.out.println("Rotated by "+ (180*rtangle/Math.PI) + " degrees");
        return result;
    }
    
    public static double[] scale(double[] input) {
        // 0->1 to min->max
        // 0->max-min, min->max 
        // scale from min to max
        Random rand = new Random();
        double scale =(MAX_SCALE-MIN_SCALE)*rand.nextDouble() + MIN_SCALE;
        double[] result = new double[784];
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                // translate 0->27 -> -13.5->13.5
                double x = j - 13.5;
                double y = i - 13.5;
                double newX = x*scale;
                double newY = y*scale;
                //compute place in array
                double newJ = newX + 13.5;
                double newI = newY + 13.5;
                result[28*i + j] = estimateValue(newI, newJ, input);
            }
        }
        System.out.println("Scaled by " + 1/scale);
        return result;
    }
    
    public static double[] transpose(double[] input) {
        // angle = 0 -> 2pi
        Random rand = new Random();
        double[] result = new double[784];
        double angle = 2*Math.PI*rand.nextDouble();
        double dsp = MAX_DISPLACEMENT*rand.nextDouble();
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                double newI = i + dsp*Math.sin(angle);
                double newJ = j + dsp*Math.cos(angle);
                result[28*i + j] = estimateValue(newI, newJ, input);
            }
        }
        System.out.println("Transposed with displacement " + dsp
                           + " and angle " + (180*angle/Math.PI) + " degrees");
        return result;
    }
}
