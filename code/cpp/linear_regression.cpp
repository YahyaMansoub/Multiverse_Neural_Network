/**
 * Linear Regression from First Principles - C++ Implementation
 * Implements gradient descent-based linear regression with MSE loss.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

class LinearRegression {
private:
    std::vector<double> w;        // Weight vector
    double b;                     // Bias term
    std::vector<double> losses;   // Training loss history
    int n_features;               // Number of features
    
    // Random number generator
    std::mt19937 gen;
    std::normal_distribution<double> dist;
    
public:
    /**
     * Constructor
     */
    LinearRegression() : b(0.0), n_features(0), gen(42), dist(0.0, 1.0) {}
    
    /**
     * Forward pass: y_hat = X @ w + b
     */
    std::vector<double> forward(const std::vector<std::vector<double>>& X) const {
        int n_samples = X.size();
        std::vector<double> y_hat(n_samples);
        
        for (int i = 0; i < n_samples; i++) {
            y_hat[i] = b;
            for (int j = 0; j < n_features; j++) {
                y_hat[i] += X[i][j] * w[j];
            }
        }
        
        return y_hat;
    }
    
    /**
     * Compute MSE loss
     */
    double compute_loss(const std::vector<double>& y_hat, 
                       const std::vector<double>& y) const {
        double sum = 0.0;
        int n_samples = y.size();
        
        for (int i = 0; i < n_samples; i++) {
            double diff = y_hat[i] - y[i];
            sum += diff * diff;
        }
        
        return sum / n_samples;
    }
    
    /**
     * Compute gradients
     */
    void compute_gradients(const std::vector<std::vector<double>>& X,
                          const std::vector<double>& y,
                          const std::vector<double>& y_hat,
                          std::vector<double>& dw,
                          double& db) const {
        int n_samples = X.size();
        
        // Initialize gradients
        std::fill(dw.begin(), dw.end(), 0.0);
        db = 0.0;
        
        // Compute gradients
        for (int i = 0; i < n_samples; i++) {
            double error = y_hat[i] - y[i];
            db += error;
            for (int j = 0; j < n_features; j++) {
                dw[j] += error * X[i][j];
            }
        }
        
        // Scale by 2/n
        double scale = 2.0 / n_samples;
        db *= scale;
        for (int j = 0; j < n_features; j++) {
            dw[j] *= scale;
        }
    }
    
    /**
     * Train the model using gradient descent
     */
    void fit(const std::vector<std::vector<double>>& X,
            const std::vector<double>& y,
            double learning_rate = 0.01,
            int epochs = 1000,
            bool verbose = true) {
        
        int n_samples = X.size();
        n_features = X[0].size();
        
        // Initialize weights randomly
        w.resize(n_features);
        for (int j = 0; j < n_features; j++) {
            w[j] = dist(gen);
        }
        b = 0.0;
        
        // Training loop
        losses.clear();
        std::vector<double> dw(n_features);
        double db;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            std::vector<double> y_hat = forward(X);
            
            // Compute loss
            double loss = compute_loss(y_hat, y);
            losses.push_back(loss);
            
            // Compute gradients
            compute_gradients(X, y, y_hat, dw, db);
            
            // Update parameters
            for (int j = 0; j < n_features; j++) {
                w[j] -= learning_rate * dw[j];
            }
            b -= learning_rate * db;
            
            // Print progress
            if (verbose && (epoch + 1) % 200 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                         << ", Loss: " << std::fixed << std::setprecision(6) 
                         << loss << std::endl;
            }
        }
        
        if (verbose) {
            std::cout << "\nTraining completed. Final loss: " 
                     << std::fixed << std::setprecision(6) 
                     << losses.back() << std::endl;
        }
    }
    
    /**
     * Make predictions
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        if (w.empty()) {
            throw std::runtime_error("Model must be fitted before making predictions.");
        }
        return forward(X);
    }
    
    /**
     * Get learned parameters
     */
    std::pair<std::vector<double>, double> get_parameters() const {
        return {w, b};
    }
    
    /**
     * Compute R² score
     */
    double score(const std::vector<std::vector<double>>& X,
                const std::vector<double>& y) const {
        std::vector<double> y_hat = predict(X);
        
        // Compute mean of y
        double y_mean = 0.0;
        for (double val : y) {
            y_mean += val;
        }
        y_mean /= y.size();
        
        // Compute SS_res and SS_tot
        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            double diff_res = y[i] - y_hat[i];
            double diff_tot = y[i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        
        return 1.0 - (ss_res / ss_tot);
    }
};

/**
 * Example usage
 */
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "Linear Regression from First Principles - C++ Implementation" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Random number generator for data
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Generate synthetic data
    int n_samples = 100;
    int n_features = 2;
    
    std::vector<double> true_w = {3.0, 2.0};
    double true_b = 1.0;
    
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(n_features));
    std::vector<double> y(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        double sum = true_b;
        for (int j = 0; j < n_features; j++) {
            X[i][j] = dist(gen);
            sum += X[i][j] * true_w[j];
        }
        y[i] = sum + 0.5 * dist(gen);
    }
    
    std::cout << "\nData shape: X=(" << n_samples << ", " << n_features 
              << "), y=(" << n_samples << ")" << std::endl;
    std::cout << "True parameters: w=[" << true_w[0] << ", " << true_w[1] 
              << "], b=" << std::fixed << std::setprecision(4) << true_b << std::endl;
    
    // Create and train model
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "Training Model..." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    LinearRegression model;
    model.fit(X, y, 0.1, 1000, true);
    
    // Get learned parameters
    auto [w_learned, b_learned] = model.get_parameters();
    std::cout << "\nLearned parameters: w=[" << std::fixed << std::setprecision(5)
              << w_learned[0] << ", " << w_learned[1] << "], b=" 
              << std::setprecision(4) << b_learned << std::endl;
    std::cout << "True parameters: w=[" << true_w[0] << ", " << true_w[1] 
              << "], b=" << true_b << std::endl;
    
    // Compute R² score
    double r2 = model.score(X, y);
    std::cout << "\nR² Score: " << std::fixed << std::setprecision(6) 
              << r2 << std::endl;
    
    // Make predictions on new data
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "Making Predictions on New Data..." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    std::vector<std::vector<double>> X_new = {
        {1.0, 2.0},
        {2.0, 3.0},
        {0.5, 1.5}
    };
    
    std::vector<double> predictions = model.predict(X_new);
    
    std::cout << "\nNew data:" << std::endl;
    for (const auto& row : X_new) {
        std::cout << "[" << row[0] << ", " << row[1] << "]" << std::endl;
    }
    
    std::cout << "\nPredictions:" << std::endl;
    for (double pred : predictions) {
        std::cout << std::fixed << std::setprecision(5) << pred << std::endl;
    }
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    return 0;
}
