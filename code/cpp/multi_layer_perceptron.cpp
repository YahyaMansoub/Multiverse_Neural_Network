/*
 * Multi-Layer Perceptron Implementation in C++
 * A neural network with multiple hidden layers for binary classification.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>

using namespace std;

// Activation function enums
enum class ActivationType {
    SIGMOID,
    RELU,
    TANH
};

// Matrix class (simplified)
class Matrix {
public:
    vector<vector<double>> data;
    int rows, cols;
    
    Matrix(int r = 0, int c = 0) : rows(r), cols(c) {
        data.resize(r, vector<double>(c, 0.0));
    }
    
    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }
    
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
    
    Matrix apply(function<double(double)> func) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }
    
    Matrix elementwise_multiply(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }
};

// Activation functions
class Activations {
public:
    static double sigmoid(double z) {
        if (z < -500) z = -500;
        if (z > 500) z = 500;
        return 1.0 / (1.0 + exp(-z));
    }
    
    static double sigmoid_derivative(double a) {
        return a * (1.0 - a);
    }
    
    static double relu(double z) {
        return max(0.0, z);
    }
    
    static double relu_derivative(double z) {
        return z > 0 ? 1.0 : 0.0;
    }
    
    static double tanh_func(double z) {
        return tanh(z);
    }
    
    static double tanh_derivative(double a) {
        return 1.0 - a * a;
    }
};

// Layer class
class Layer {
public:
    Matrix W, b;
    Matrix z, a, x;
    ActivationType activation;
    
    Layer(int input_size, int output_size, ActivationType act) 
        : W(output_size, input_size), b(output_size, 1), activation(act) {
        
        // Xavier/Glorot initialization
        random_device rd;
        mt19937 gen(rd());
        double limit = sqrt(6.0 / (input_size + output_size));
        uniform_real_distribution<> dis(-limit, limit);
        
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W.data[i][j] = dis(gen);
            }
        }
    }
    
    Matrix forward(const Matrix& input) {
        x = input;
        
        // z = W * x + b
        z = W * x;
        for (int i = 0; i < z.rows; i++) {
            for (int j = 0; j < z.cols; j++) {
                z.data[i][j] += b.data[i][0];
            }
        }
        
        // Apply activation
        switch (activation) {
            case ActivationType::SIGMOID:
                a = z.apply(Activations::sigmoid);
                break;
            case ActivationType::RELU:
                a = z.apply(Activations::relu);
                break;
            case ActivationType::TANH:
                a = z.apply(Activations::tanh_func);
                break;
        }
        
        return a;
    }
    
    tuple<Matrix, Matrix, Matrix> backward(const Matrix& delta_next) {
        int batch_size = x.cols;
        
        // Apply activation derivative
        Matrix delta(delta_next.rows, delta_next.cols);
        
        if (activation == ActivationType::SIGMOID || activation == ActivationType::TANH) {
            auto deriv_func = (activation == ActivationType::SIGMOID) 
                ? Activations::sigmoid_derivative 
                : Activations::tanh_derivative;
            
            Matrix deriv = a.apply(deriv_func);
            delta = delta_next.elementwise_multiply(deriv);
        } else {  // RELU
            Matrix deriv = z.apply(Activations::relu_derivative);
            delta = delta_next.elementwise_multiply(deriv);
        }
        
        // Compute gradients
        Matrix dW = delta * x.transpose();
        for (int i = 0; i < dW.rows; i++) {
            for (int j = 0; j < dW.cols; j++) {
                dW.data[i][j] /= batch_size;
            }
        }
        
        Matrix db(delta.rows, 1);
        for (int i = 0; i < delta.rows; i++) {
            double sum = 0;
            for (int j = 0; j < delta.cols; j++) {
                sum += delta.data[i][j];
            }
            db.data[i][0] = sum / batch_size;
        }
        
        // Gradient for previous layer
        Matrix delta_prev = W.transpose() * delta;
        
        return make_tuple(delta_prev, dW, db);
    }
};

// MLP class
class MLP {
private:
    vector<Layer> layers;
    vector<double> losses;
    
public:
    MLP(vector<int> layer_sizes, vector<ActivationType> activations) {
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i+1], activations[i]);
        }
    }
    
    Matrix forward(const Matrix& X) {
        Matrix a = X;
        for (auto& layer : layers) {
            a = layer.forward(a);
        }
        return a;
    }
    
    vector<tuple<Matrix, Matrix>> backward(const Matrix& y_true, const Matrix& y_pred) {
        // Initial gradient
        Matrix delta = y_pred - y_true;
        
        vector<tuple<Matrix, Matrix>> gradients;
        
        // Backpropagate
        for (int i = layers.size() - 1; i >= 0; i--) {
            Matrix delta_prev, dW, db;
            
            if (i == (int)layers.size() - 1) {
                // Output layer
                int batch_size = layers[i].x.cols;
                dW = delta * layers[i].x.transpose();
                for (int r = 0; r < dW.rows; r++) {
                    for (int c = 0; c < dW.cols; c++) {
                        dW.data[r][c] /= batch_size;
                    }
                }
                
                db = Matrix(delta.rows, 1);
                for (int r = 0; r < delta.rows; r++) {
                    double sum = 0;
                    for (int c = 0; c < delta.cols; c++) {
                        sum += delta.data[r][c];
                    }
                    db.data[r][0] = sum / batch_size;
                }
                
                delta_prev = layers[i].W.transpose() * delta;
            } else {
                // Hidden layers
                tie(delta_prev, dW, db) = layers[i].backward(delta);
            }
            
            gradients.insert(gradients.begin(), make_tuple(dW, db));
            delta = delta_prev;
        }
        
        return gradients;
    }
    
    void update_parameters(const vector<tuple<Matrix, Matrix>>& gradients, double learning_rate) {
        for (size_t i = 0; i < layers.size(); i++) {
            Matrix dW = get<0>(gradients[i]);
            Matrix db = get<1>(gradients[i]);
            
            for (int r = 0; r < layers[i].W.rows; r++) {
                for (int c = 0; c < layers[i].W.cols; c++) {
                    layers[i].W.data[r][c] -= learning_rate * dW.data[r][c];
                }
            }
            
            for (int r = 0; r < layers[i].b.rows; r++) {
                layers[i].b.data[r][0] -= learning_rate * db.data[r][0];
            }
        }
    }
    
    double binary_cross_entropy(const Matrix& y_true, const Matrix& y_pred) {
        double loss = 0.0;
        double epsilon = 1e-15;
        int n = y_true.cols;
        
        for (int j = 0; j < n; j++) {
            double pred = max(epsilon, min(1.0 - epsilon, y_pred.data[0][j]));
            double true_val = y_true.data[0][j];
            loss += true_val * log(pred) + (1.0 - true_val) * log(1.0 - pred);
        }
        
        return -loss / n;
    }
    
    void fit(const Matrix& X, const vector<double>& y, double learning_rate = 0.1, 
             int epochs = 1000, bool verbose = true) {
        // Transpose input
        Matrix X_T = X.transpose();
        Matrix y_T(1, y.size());
        for (size_t i = 0; i < y.size(); i++) {
            y_T.data[0][i] = y[i];
        }
        
        losses.clear();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            Matrix y_pred = forward(X_T);
            
            // Compute loss
            double loss = binary_cross_entropy(y_T, y_pred);
            losses.push_back(loss);
            
            // Backward pass
            auto gradients = backward(y_T, y_pred);
            
            // Update parameters
            update_parameters(gradients, learning_rate);
            
            // Print progress
            if (verbose && (epoch + 1) % 1000 == 0) {
                cout << "Epoch " << epoch + 1 << "/" << epochs 
                     << ", Loss: " << fixed << setprecision(6) << loss << endl;
            }
        }
        
        if (verbose) {
            cout << "\nTraining completed. Final loss: " << fixed << setprecision(6) 
                 << losses.back() << endl;
        }
    }
    
    Matrix predict_proba(const Matrix& X) {
        Matrix X_T = X.transpose();
        Matrix result = forward(X_T);
        return result.transpose();
    }
    
    vector<int> predict(const Matrix& X, double threshold = 0.5) {
        Matrix proba = predict_proba(X);
        vector<int> predictions;
        
        for (int i = 0; i < proba.rows; i++) {
            predictions.push_back(proba.data[i][0] >= threshold ? 1 : 0);
        }
        
        return predictions;
    }
    
    double score(const Matrix& X, const vector<double>& y) {
        auto predictions = predict(X);
        int correct = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (predictions[i] == (int)y[i]) correct++;
        }
        return (double)correct / y.size();
    }
};

int main() {
    cout << "Training MLP on XOR problem\n" << endl;
    
    // XOR dataset
    Matrix X(4, 2);
    X.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    vector<double> y = {0, 1, 1, 0};
    
    // Create and train MLP
    MLP mlp({2, 4, 1}, {ActivationType::TANH, ActivationType::SIGMOID});
    mlp.fit(X, y, 0.5, 5000, true);
    
    // Test predictions
    cout << "\n--- Predictions ---" << endl;
    auto predictions = mlp.predict(X);
    auto probas = mlp.predict_proba(X);
    
    for (int i = 0; i < 4; i++) {
        cout << "Input: [" << X.data[i][0] << ", " << X.data[i][1] << "], "
             << "Predicted: " << predictions[i] << ", "
             << "Probability: " << fixed << setprecision(4) << probas.data[i][0] << ", "
             << "Actual: " << (int)y[i] << endl;
    }
    
    cout << "\nAccuracy: " << fixed << setprecision(2) << mlp.score(X, y) * 100 << "%" << endl;
    
    return 0;
}
