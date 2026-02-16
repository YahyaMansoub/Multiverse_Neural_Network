/*
 * Multi-Layer Perceptron Implementation in Java
 * A neural network with multiple hidden layers for binary classification.
 */

import java.util.*;
import java.util.function.Function;

// Activation function enums
enum ActivationType {
    SIGMOID, RELU, TANH
}

// Matrix class
class Matrix {
    private double[][] data;
    private int rows;
    private int cols;
    
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }
    
    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }
    
    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double[][] getData() { return data; }
    
    public double get(int i, int j) { return data[i][j]; }
    public void set(int i, int j, double value) { data[i][j] = value; }
    
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    public Matrix add(Matrix other) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix subtract(Matrix other) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix apply(Function<Double, Double> func) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = func.apply(this.data[i][j]);
            }
        }
        return result;
    }
    
    public Matrix elementwiseMultiply(Matrix other) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix copy() {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            System.arraycopy(this.data[i], 0, result.data[i], 0, this.cols);
        }
        return result;
    }
}

// Activation functions class
class Activations {
    public static double sigmoid(double z) {
        if (z < -500) z = -500;
        if (z > 500) z = 500;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    public static double sigmoidDerivative(double a) {
        return a * (1.0 - a);
    }
    
    public static double relu(double z) {
        return Math.max(0, z);
    }
    
    public static double reluDerivative(double z) {
        return z > 0 ? 1.0 : 0.0;
    }
    
    public static double tanh(double z) {
        return Math.tanh(z);
    }
    
    public static double tanhDerivative(double a) {
        return 1.0 - a * a;
    }
}

// Layer class
class Layer {
    private Matrix W;
    private Matrix b;
    private Matrix z;
    private Matrix a;
    private Matrix x;
    private ActivationType activation;
    
    public Layer(int inputSize, int outputSize, ActivationType activation) {
        this.W = new Matrix(outputSize, inputSize);
        this.b = new Matrix(outputSize, 1);
        this.activation = activation;
        
        // Xavier/Glorot initialization
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        Random rand = new Random();
        
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W.set(i, j, rand.nextDouble() * 2 * limit - limit);
            }
        }
    }
    
    public Matrix forward(Matrix input) {
        this.x = input.copy();
        
        // z = W * x + b
        this.z = W.multiply(input);
        for (int i = 0; i < z.getRows(); i++) {
            for (int j = 0; j < z.getCols(); j++) {
                z.set(i, j, z.get(i, j) + b.get(i, 0));
            }
        }
        
        // Apply activation
        switch (activation) {
            case SIGMOID:
                this.a = z.apply(Activations::sigmoid);
                break;
            case RELU:
                this.a = z.apply(Activations::relu);
                break;
            case TANH:
                this.a = z.apply(Activations::tanh);
                break;
        }
        
        return this.a;
    }
    
    public LayerGradients backward(Matrix deltaNext) {
        int batchSize = x.getCols();
        
        // Apply activation derivative
        Matrix delta;
        if (activation == ActivationType.SIGMOID || activation == ActivationType.TANH) {
            Function<Double, Double> derivFunc = (activation == ActivationType.SIGMOID) 
                ? Activations::sigmoidDerivative 
                : Activations::tanhDerivative;
            
            Matrix deriv = a.apply(derivFunc);
            delta = deltaNext.elementwiseMultiply(deriv);
        } else { // RELU
            Matrix deriv = z.apply(Activations::reluDerivative);
            delta = deltaNext.elementwiseMultiply(deriv);
        }
        
        // Compute gradients
        Matrix dW = delta.multiply(x.transpose());
        for (int i = 0; i < dW.getRows(); i++) {
            for (int j = 0; j < dW.getCols(); j++) {
                dW.set(i, j, dW.get(i, j) / batchSize);
            }
        }
        
        Matrix db = new Matrix(delta.getRows(), 1);
        for (int i = 0; i < delta.getRows(); i++) {
            double sum = 0;
            for (int j = 0; j < delta.getCols(); j++) {
                sum += delta.get(i, j);
            }
            db.set(i, 0, sum / batchSize);
        }
        
        // Gradient for previous layer
        Matrix deltaPrev = W.transpose().multiply(delta);
        
        return new LayerGradients(deltaPrev, dW, db);
    }
    
    public Matrix getW() { return W; }
    public Matrix getB() { return b; }
    public Matrix getX() { return x; }
    public void updateW(Matrix dW, double learningRate) {
        for (int i = 0; i < W.getRows(); i++) {
            for (int j = 0; j < W.getCols(); j++) {
                W.set(i, j, W.get(i, j) - learningRate * dW.get(i, j));
            }
        }
    }
    
    public void updateB(Matrix db, double learningRate) {
        for (int i = 0; i < b.getRows(); i++) {
            b.set(i, 0, b.get(i, 0) - learningRate * db.get(i, 0));
        }
    }
}

// Helper class for layer gradients
class LayerGradients {
    Matrix deltaPrev;
    Matrix dW;
    Matrix db;
    
    public LayerGradients(Matrix deltaPrev, Matrix dW, Matrix db) {
        this.deltaPrev = deltaPrev;
        this.dW = dW;
        this.db = db;
    }
}

// MLP class
class MLP {
    private List<Layer> layers;
    private List<Double> losses;
    
    public MLP(int[] layerSizes, ActivationType[] activations) {
        this.layers = new ArrayList<>();
        this.losses = new ArrayList<>();
        
        for (int i = 0; i < layerSizes.length - 1; i++) {
            layers.add(new Layer(layerSizes[i], layerSizes[i+1], activations[i]));
        }
    }
    
    public Matrix forward(Matrix X) {
        Matrix a = X;
        for (Layer layer : layers) {
            a = layer.forward(a);
        }
        return a;
    }
    
    public double binaryCrossEntropy(Matrix yTrue, Matrix yPred) {
        double loss = 0.0;
        double epsilon = 1e-15;
        int n = yTrue.getCols();
        
        for (int j = 0; j < n; j++) {
            double pred = Math.max(epsilon, Math.min(1.0 - epsilon, yPred.get(0, j)));
            double trueVal = yTrue.get(0, j);
            loss += trueVal * Math.log(pred) + (1.0 - trueVal) * Math.log(1.0 - pred);
        }
        
        return -loss / n;
    }
    
    public List<LayerGradients> backward(Matrix yTrue, Matrix yPred) {
        // Initial gradient
        Matrix delta = yPred.subtract(yTrue);
        
        List<LayerGradients> gradients = new ArrayList<>();
        
        // Backpropagate
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            LayerGradients grad;
            
            if (i == layers.size() - 1) {
                // Output layer
                int batchSize = layer.getX().getCols();
                Matrix dW = delta.multiply(layer.getX().transpose());
                for (int r = 0; r < dW.getRows(); r++) {
                    for (int c = 0; c < dW.getCols(); c++) {
                        dW.set(r, c, dW.get(r, c) / batchSize);
                    }
                }
                
                Matrix db = new Matrix(delta.getRows(), 1);
                for (int r = 0; r < delta.getRows(); r++) {
                    double sum = 0;
                    for (int c = 0; c < delta.getCols(); c++) {
                        sum += delta.get(r, c);
                    }
                    db.set(r, 0, sum / batchSize);
                }
                
                Matrix deltaPrev = layer.getW().transpose().multiply(delta);
                grad = new LayerGradients(deltaPrev, dW, db);
            } else {
                // Hidden layers
                grad = layer.backward(delta);
            }
            
            gradients.add(0, grad);
            delta = grad.deltaPrev;
        }
        
        return gradients;
    }
    
    public void updateParameters(List<LayerGradients> gradients, double learningRate) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            LayerGradients grad = gradients.get(i);
            layer.updateW(grad.dW, learningRate);
            layer.updateB(grad.db, learningRate);
        }
    }
    
    public void fit(Matrix X, double[] y, double learningRate, int epochs, boolean verbose) {
        // Transpose input
        Matrix XT = X.transpose();
        Matrix yT = new Matrix(1, y.length);
        for (int i = 0; i < y.length; i++) {
            yT.set(0, i, y[i]);
        }
        
        losses.clear();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            Matrix yPred = forward(XT);
            
            // Compute loss
            double loss = binaryCrossEntropy(yT, yPred);
            losses.add(loss);
            
            // Backward pass
            List<LayerGradients> gradients = backward(yT, yPred);
            
            // Update parameters
            updateParameters(gradients, learningRate);
            
            // Print progress
            if (verbose && (epoch + 1) % 1000 == 0) {
                System.out.printf("Epoch %d/%d, Loss: %.6f%n", epoch + 1, epochs, loss);
            }
        }
        
        if (verbose) {
            System.out.printf("%nTraining completed. Final loss: %.6f%n", losses.get(losses.size() - 1));
        }
    }
    
    public Matrix predictProba(Matrix X) {
        Matrix XT = X.transpose();
        Matrix result = forward(XT);
        return result.transpose();
    }
    
    public int[] predict(Matrix X, double threshold) {
        Matrix proba = predictProba(X);
        int[] predictions = new int[proba.getRows()];
        
        for (int i = 0; i < proba.getRows(); i++) {
            predictions[i] = proba.get(i, 0) >= threshold ? 1 : 0;
        }
        
        return predictions;
    }
    
    public double score(Matrix X, double[] y) {
        int[] predictions = predict(X, 0.5);
        int correct = 0;
        
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == (int)y[i]) {
                correct++;
            }
        }
        
        return (double)correct / y.length;
    }
}

// Main class
public class MultiLayerPerceptron {
    public static void main(String[] args) {
        System.out.println("Training MLP on XOR problem\n");
        
        // XOR dataset
        double[][] XData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        Matrix X = new Matrix(XData);
        double[] y = {0, 1, 1, 0};
        
        // Create and train MLP
        int[] layerSizes = {2, 4, 1};
        ActivationType[] activations = {ActivationType.TANH, ActivationType.SIGMOID};
        
        MLP mlp = new MLP(layerSizes, activations);
        mlp.fit(X, y, 0.5, 5000, true);
        
        // Test predictions
        System.out.println("\n--- Predictions ---");
        int[] predictions = mlp.predict(X, 0.5);
        Matrix probas = mlp.predictProba(X);
        
        for (int i = 0; i < 4; i++) {
            System.out.printf("Input: [%.0f, %.0f], Predicted: %d, Probability: %.4f, Actual: %.0f%n",
                XData[i][0], XData[i][1], predictions[i], probas.get(i, 0), y[i]);
        }
        
        System.out.printf("%nAccuracy: %.0f%%%n", mlp.score(X, y) * 100);
    }
}
