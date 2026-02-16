/**
 * Linear Regression from First Principles - C Implementation
 * Implements gradient descent-based linear regression with MSE loss.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    double* w;          // Weight vector
    double b;           // Bias term
    int n_features;     // Number of features
    double* losses;     // Training loss history
    int n_losses;       // Number of recorded losses
} LinearRegression;

/**
 * Initialize a linear regression model
 */
LinearRegression* create_model(int n_features) {
    LinearRegression* model = (LinearRegression*)malloc(sizeof(LinearRegression));
    model->w = (double*)malloc(n_features * sizeof(double));
    model->b = 0.0;
    model->n_features = n_features;
    model->losses = NULL;
    model->n_losses = 0;
    return model;
}

/**
 * Free model memory
 */
void free_model(LinearRegression* model) {
    if (model) {
        free(model->w);
        free(model->losses);
        free(model);
    }
}

/**
 * Random number generator for weight initialization
 */
double randn() {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * Forward pass: y_hat = X @ w + b
 */
void forward(double* X, double* w, double b, int n_samples, int n_features, double* y_hat) {
    for (int i = 0; i < n_samples; i++) {
        y_hat[i] = b;
        for (int j = 0; j < n_features; j++) {
            y_hat[i] += X[i * n_features + j] * w[j];
        }
    }
}

/**
 * Compute MSE loss
 */
double compute_mse(double* y_hat, double* y, int n_samples) {
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double diff = y_hat[i] - y[i];
        sum += diff * diff;
    }
    return sum / n_samples;
}

/**
 * Compute gradients
 */
void compute_gradients(double* X, double* y, double* y_hat, int n_samples, 
                       int n_features, double* dw, double* db) {
    // Initialize gradients to zero
    for (int j = 0; j < n_features; j++) {
        dw[j] = 0.0;
    }
    *db = 0.0;
    
    // Compute error and accumulate gradients
    for (int i = 0; i < n_samples; i++) {
        double error = y_hat[i] - y[i];
        *db += error;
        for (int j = 0; j < n_features; j++) {
            dw[j] += error * X[i * n_features + j];
        }
    }
    
    // Scale by 2/n
    double scale = 2.0 / n_samples;
    *db *= scale;
    for (int j = 0; j < n_features; j++) {
        dw[j] *= scale;
    }
}

/**
 * Train the model using gradient descent
 */
void fit(LinearRegression* model, double* X, double* y, int n_samples, 
         double learning_rate, int epochs, int verbose) {
    
    // Initialize weights randomly
    srand(42);
    for (int j = 0; j < model->n_features; j++) {
        model->w[j] = randn();
    }
    model->b = 0.0;
    
    // Allocate memory for losses
    model->losses = (double*)malloc(epochs * sizeof(double));
    model->n_losses = epochs;
    
    // Allocate temporary arrays
    double* y_hat = (double*)malloc(n_samples * sizeof(double));
    double* dw = (double*)malloc(model->n_features * sizeof(double));
    double db;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        forward(X, model->w, model->b, n_samples, model->n_features, y_hat);
        
        // Compute loss
        double loss = compute_mse(y_hat, y, n_samples);
        model->losses[epoch] = loss;
        
        // Compute gradients
        compute_gradients(X, y, y_hat, n_samples, model->n_features, dw, &db);
        
        // Update parameters
        for (int j = 0; j < model->n_features; j++) {
            model->w[j] -= learning_rate * dw[j];
        }
        model->b -= learning_rate * db;
        
        // Print progress
        if (verbose && (epoch + 1) % 200 == 0) {
            printf("Epoch %d/%d, Loss: %.6f\n", epoch + 1, epochs, loss);
        }
    }
    
    if (verbose) {
        printf("\nTraining completed. Final loss: %.6f\n", model->losses[epochs - 1]);
    }
    
    // Free temporary arrays
    free(y_hat);
    free(dw);
}

/**
 * Make predictions
 */
void predict(LinearRegression* model, double* X, int n_samples, double* predictions) {
    forward(X, model->w, model->b, n_samples, model->n_features, predictions);
}

/**
 * Compute R² score
 */
double score(LinearRegression* model, double* X, double* y, int n_samples) {
    double* y_hat = (double*)malloc(n_samples * sizeof(double));
    predict(model, X, n_samples, y_hat);
    
    // Compute mean of y
    double y_mean = 0.0;
    for (int i = 0; i < n_samples; i++) {
        y_mean += y[i];
    }
    y_mean /= n_samples;
    
    // Compute SS_res and SS_tot
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double diff_res = y[i] - y_hat[i];
        double diff_tot = y[i] - y_mean;
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }
    
    free(y_hat);
    return 1.0 - (ss_res / ss_tot);
}

/**
 * Example usage
 */
int main() {
    printf("============================================================\n");
    printf("Linear Regression from First Principles - C Implementation\n");
    printf("============================================================\n");
    
    // Set random seed
    srand(42);
    
    // Generate synthetic data
    int n_samples = 100;
    int n_features = 2;
    
    double true_w[] = {3.0, 2.0};
    double true_b = 1.0;
    
    // Allocate memory for data
    double* X = (double*)malloc(n_samples * n_features * sizeof(double));
    double* y = (double*)malloc(n_samples * sizeof(double));
    
    // Generate data
    for (int i = 0; i < n_samples; i++) {
        double sum = true_b;
        for (int j = 0; j < n_features; j++) {
            X[i * n_features + j] = randn();
            sum += X[i * n_features + j] * true_w[j];
        }
        y[i] = sum + 0.5 * randn();
    }
    
    printf("\nData shape: X=(%d, %d), y=(%d)\n", n_samples, n_features, n_samples);
    printf("True parameters: w=[%.1f, %.1f], b=%.4f\n", true_w[0], true_w[1], true_b);
    
    // Create and train model
    printf("\n------------------------------------------------------------\n");
    printf("Training Model...\n");
    printf("------------------------------------------------------------\n");
    
    LinearRegression* model = create_model(n_features);
    fit(model, X, y, n_samples, 0.1, 1000, 1);
    
    // Print learned parameters
    printf("\nLearned parameters: w=[%.5f, %.5f], b=%.4f\n", 
           model->w[0], model->w[1], model->b);
    printf("True parameters: w=[%.1f, %.1f], b=%.4f\n", true_w[0], true_w[1], true_b);
    
    // Compute R² score
    double r2 = score(model, X, y, n_samples);
    printf("\nR² Score: %.6f\n", r2);
    
    // Make predictions on new data
    printf("\n------------------------------------------------------------\n");
    printf("Making Predictions on New Data...\n");
    printf("------------------------------------------------------------\n");
    
    double X_new[] = {1.0, 2.0, 2.0, 3.0, 0.5, 1.5};
    int n_new = 3;
    double* predictions = (double*)malloc(n_new * sizeof(double));
    
    predict(model, X_new, n_new, predictions);
    
    printf("\nNew data:\n");
    for (int i = 0; i < n_new; i++) {
        printf("[%.1f, %.1f]\n", X_new[i * 2], X_new[i * 2 + 1]);
    }
    
    printf("\nPredictions:\n");
    for (int i = 0; i < n_new; i++) {
        printf("%.5f\n", predictions[i]);
    }
    
    printf("\n============================================================\n");
    printf("Example completed successfully!\n");
    printf("============================================================\n");
    
    // Cleanup
    free(X);
    free(y);
    free(predictions);
    free_model(model);
    
    return 0;
}
