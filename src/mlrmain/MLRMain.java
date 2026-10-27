/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package mlrmain;

/**
 *
 * @author abdul
 */
public class MLRMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        double[][] X = {
            {1, 6, 1},
            {2, 8, 2},
            {3, 7, 1},
            {4, 6, 3},
            {5, 8, 2},
            {6, 7, 4},
            {7, 9, 3},
            {8, 6, 5}
        };

        double[] y = {
            45, 55, 58, 65, 72, 80, 85, 90
        };

        String[] featureNames = {
            "Study Hours",
            "Sleep Hours",
            "Practice Exams"
        };

        double[][] scaled = MultipleLinearRegression.standardize(X);
        
        MultipleLinearRegression.gradDescent(scaled, y);
        double[] newStudent = {5, 7, 3};

        double predictedScore = MultipleLinearRegression.predictNew(newStudent);

        System.out.printf(
            "\nPredicted Score for New Student: %.2f%n",
            predictedScore
        );
        
    }
    
}
