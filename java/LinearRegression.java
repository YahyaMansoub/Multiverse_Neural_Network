/**
 * Linear Regression from First Principles - Java Implementation
 * Implements gradient descent-based linear regression with MSE loss.
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LinearRegression {
    private double[] w;              // Weight vector
    private double b;                // Bias term
    private List<Double> losses;     // Training loss history
    private int nFeatures;           // Number of features
    private Random random;
    
    /**
     * Constructor
     */
    public LinearRegression() {
        this.b = 0.0;
        this.losses = new ArrayList<>();
        this.random = new Random(42);
    }
    
    /**
     * Forward pass: y_hat = X @ w + b
     */
    private double[] forward(double[][] X) {
        int nSamples = X.length;
        double[] yHat = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            yHat[i] = b;
            for (int j = 0; j < nFeatures; j++) {
                yHat[i] += X[i][j] * w[j];
            }
        }
        
        return yHat;
    }
    
    /**
     * Compute MSE loss
     */
    private double computeLoss(double[] yHat, double[] y) {
        double sum = 0.0;
        int nSamples = y.length;
        
        for (int i = 0; i < nSamples; i++) {
            double diff = yHat[i] - y[i];
            sum += diff * diff;
        }
        
        return sum / nSamples;
    }
    
    /**
     * Compute gradients
     */
    private GradientResult computeGradients(double[][] X, double[] y, double[] yHat) {
        int nSamples = X.length;
        double[] dw = new double[nFeatures];
        double db = 0.0;
        
        // Compute gradients
        for (int i = 0; i < nSamples; i++) {
            double error = yHat[i] - y[i];
            db += error;
            for (int j = 0; j < nFeatures; j++) {
                dw[j] += error * X[i][j];
            }
        }
        
        // Scale by 2/n
        double scale = 2.0 / nSamples;
        db *= scale;
        for (int j = 0; j < nFeatures; j++) {
            dw[j] *= scale;
        }
        
        return new GradientResult(dw, db);
    }
    
    /**
     * Train the model using gradient descent
     */
    public void fit(double[][] X, double[] y, double learningRate, int epochs, boolean verbose) {
        int nSamples = X.length;
        nFeatures = X[0].length;
        
        // Initialize weights randomly
        w = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            w[j] = random.nextGaussian();
        }
        b = 0.0;
        
        // Training loop
        losses.clear();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            double[] yHat = forward(X);
            
            // Compute loss
            double loss = computeLoss(yHat, y);
            losses.add(loss);
            
            // Compute gradients
            GradientResult gradients = computeGradients(X, y, yHat);
            
            // Update parameters
            for (int j = 0; j < nFeatures; j++) {
                w[j] -= learningRate * gradients.dw[j];
            }
            b -= learningRate * gradients.db;
            
            // Print progress
            if (verbose && (epoch + 1) % 200 == 0) {
                System.out.printf("Epoch %d/%d, Loss: %.6f%n", epoch + 1, epochs, loss);
            }
        }
        
        if (verbose) {
            System.out.printf("%nTraining completed. Final loss: %.6f%n", 
                            losses.get(losses.size() - 1));
        }
    }
    
    /**
     * Make predictions
     */
    public double[] predict(double[][] X) {
        if (w == null) {
            throw new IllegalStateException("Model must be fitted before making predictions.");
        }
        return forward(X);
    }
    
    /**
     * Get learned parameters
     */
    public Parameters getParameters() {
        return new Parameters(w, b);
    }
    
    /**
     * Compute R² score
     */
    public double score(double[][] X, double[] y) {
        double[] yHat = predict(X);
        
        // Compute mean of y
        double yMean = 0.0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= y.length;
        
        // Compute SS_res and SS_tot
        double ssRes = 0.0, ssTot = 0.0;
        for (int i = 0; i < y.length; i++) {
            double diffRes = y[i] - yHat[i];
            double diffTot = y[i] - yMean;
            ssRes += diffRes * diffRes;
            ssTot += diffTot * diffTot;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Helper class for gradient results
     */
    private static class GradientResult {
        double[] dw;
        double db;
        
        GradientResult(double[] dw, double db) {
            this.dw = dw;
            this.db = db;
        }
    }
    
    /**
     * Helper class for parameters
     */
    public static class Parameters {
        public double[] w;
        public double b;
        
        Parameters(double[] w, double b) {
            this.w = w;
            this.b = b;
        }
    }
    
    /**
     * Example usage
     */
    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("Linear Regression from First Principles - Java Implementation");
        System.out.println("============================================================");
        
        // Random number generator for data
        Random random = new Random(42);
        
        // Generate synthetic data
        int nSamples = 100;
        int nFeatures = 2;
        
        double[] trueW = {3.0, 2.0};
        double trueB = 1.0;
        
        double[][] X = new double[nSamples][nFeatures];
        double[] y = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            double sum = trueB;
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = random.nextGaussian();
                sum += X[i][j] * trueW[j];
            }
            y[i] = sum + 0.5 * random.nextGaussian();
        }
        
        System.out.printf("%nData shape: X=(%d, %d), y=(%d)%n", nSamples, nFeatures, nSamples);
        System.out.printf("True parameters: w=[%.1f, %.1f], b=%.4f%n", trueW[0], trueW[1], trueB);
        
        // Create and train model
        System.out.println("\n------------------------------------------------------------");
        System.out.println("Training Model...");
        System.out.println("------------------------------------------------------------");
        
        LinearRegression model = new LinearRegression();
        model.fit(X, y, 0.1, 1000, true);
        
        // Get learned parameters
        Parameters params = model.getParameters();
        System.out.printf("%nLearned parameters: w=[%.5f, %.5f], b=%.4f%n", 
                         params.w[0], params.w[1], params.b);
        System.out.printf("True parameters: w=[%.1f, %.1f], b=%.4f%n", trueW[0], trueW[1], trueB);
        
        // Compute R² score
        double r2 = model.score(X, y);
        System.out.printf("%nR² Score: %.6f%n", r2);
        
        // Make predictions on new data
        System.out.println("\n------------------------------------------------------------");
        System.out.println("Making Predictions on New Data...");
        System.out.println("------------------------------------------------------------");
        
        double[][] XNew = {
            {1.0, 2.0},
            {2.0, 3.0},
            {0.5, 1.5}
        };
        
        double[] predictions = model.predict(XNew);
        
        System.out.println("\nNew data:");
        for (double[] row : XNew) {
            System.out.printf("[%.1f, %.1f]%n", row[0], row[1]);
        }
        
        System.out.println("\nPredictions:");
        for (double pred : predictions) {
            System.out.printf("%.5f%n", pred);
        }
        
        System.out.println("\n============================================================");
        System.out.println("Example completed successfully!");
        System.out.println("============================================================");
    }
}
