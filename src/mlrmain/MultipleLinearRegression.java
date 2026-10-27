/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package mlrmain;

/**
 *
 * @author abdul
 */
public class MultipleLinearRegression {
    static double[] weights;
    static double bias = 0.0;
    static double learningRate= 0.01;
    static int epochs = 10000;
    
    static double[] means;
    static double[] stds;
    
    public static double[][] standardize(double[][] X) {
        int exN =  X.length;
        int features =  X[0].length;
        
        means = new double[features];
        stds = new double[features];
        double[][] scaled = new double[exN][features];
        
        for (int i = 0; i < features; i++) {
            //calculating the mean
            for (int j = 0; j < exN; j++) {
                means[i] +=  X[j][i];
            }
            means[i] /= exN;
            
            //calculating the standard deviations
            for (int j = 0; j < exN; j++) {
                stds[i] +=  Math.pow(X[j][i]-means[i], 2);
            }
            stds[i] = Math.sqrt(stds[i] / exN);
            
            if (stds[i] == 0) stds[i] = 1;
            
            //standardizing the whole thing
            for (int j = 0; j < exN; j++) {
                scaled[j][i] =  (X[j][i] - means[i]) / stds[i];
            }
        }
        
        return scaled;
    }
    
    public static double predict(double[] X) {
        double prediction = bias;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * X[i];
        }
        return prediction;
    }
    
    public static double computeCost(double[][] X, double[] Y) {
        double cost = 0.0;
        
        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(predict(X[i]) - Y[i], 2);
        }
        
        cost /= 2*X.length;
        return cost;
        
    }
    
    public static void gradDescent(double[][] X, double[] Y) {
        weights = new double[X[0].length];
        bias = 0.0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] dw = new double[X[0].length];
            double db = 0;
            
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = prediction - Y[i];

                for ( int j = 0; j < X[0].length; j++) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }
            
            //updating weights and bias
            for (int i = 0; i<X[0].length; i++) {
                weights[i] = weights[i] - learningRate * (dw[i]/X.length);
            }
            
            bias = bias - learningRate*(db/X.length);

            //print training progress
            if (epoch % 1000 == 0) {

                double cost = computeCost(X, Y);

                System.out.printf(
                    "Epoch %d | Cost: %.4f%n",
                    epoch, cost
                );
            }

        }
        
    }
    
    //predict new data
    public static double predictNew(double[] input) {
        double[] scaled = new double[input.length];
        for (int i = 0; i<input.length; i++) {
            scaled[i] = (input[i] - means[i])/stds[i];
        }
        return predict(scaled);
    }
}   
