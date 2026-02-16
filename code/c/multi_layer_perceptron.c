/*
 * Multi-Layer Perceptron Implementation in C
 * A neural network with multiple hidden layers for binary classification.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 100

// Activation function enums
typedef enum {
    SIGMOID,
    RELU,
    TANH_ACTIVATION
} ActivationType;

// Matrix structure
typedef struct {
    double** data;
    int rows;
    int cols;
} Matrix;

// Layer structure
typedef struct {
    Matrix W;           // Weights
    Matrix b;           // Biases
    Matrix z;           // Pre-activation
    Matrix a;           // Activation
    Matrix x;           // Input (cached for backprop)
    ActivationType activation;
} Layer;

// MLP structure
typedef struct {
    Layer layers[MAX_LAYERS];
    int num_layers;
    double* losses;
    int num_losses;
    int max_losses;
} MLP;

// Matrix operations
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)calloc(cols, sizeof(double));
    }
    return m;
}

void free_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    m->data = NULL;
}

void matrix_multiply(Matrix* result, Matrix* A, Matrix* B) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < A->cols; k++) {
                result->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
}

void matrix_add(Matrix* result, Matrix* A, Matrix* B) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            result->data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
}

void matrix_transpose(Matrix* result, Matrix* A) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            result->data[j][i] = A->data[i][j];
        }
    }
}

void matrix_copy(Matrix* dest, Matrix* src) {
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
}

// Activation functions
double sigmoid_func(double z) {
    if (z < -500) z = -500;
    if (z > 500) z = 500;
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_derivative_func(double a) {
    return a * (1.0 - a);
}

double relu_func(double z) {
    return (z > 0) ? z : 0;
}

double relu_derivative_func(double z) {
    return (z > 0) ? 1.0 : 0.0;
}

double tanh_func(double z) {
    return tanh(z);
}

double tanh_derivative_func(double a) {
    return 1.0 - a * a;
}

void apply_activation(Matrix* output, Matrix* input, ActivationType type) {
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            switch (type) {
                case SIGMOID:
                    output->data[i][j] = sigmoid_func(input->data[i][j]);
                    break;
                case RELU:
                    output->data[i][j] = relu_func(input->data[i][j]);
                    break;
                case TANH_ACTIVATION:
                    output->data[i][j] = tanh_func(input->data[i][j]);
                    break;
            }
        }
    }
}

// Layer initialization
void init_layer(Layer* layer, int input_size, int output_size, ActivationType activation) {
    layer->W = create_matrix(output_size, input_size);
    layer->b = create_matrix(output_size, 1);
    layer->activation = activation;
    
    // Xavier/Glorot initialization
    double limit = sqrt(6.0 / (input_size + output_size));
    srand(time(NULL) + rand());
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer->W.data[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }
}

void free_layer(Layer* layer) {
    free_matrix(&layer->W);
    free_matrix(&layer->b);
    if (layer->z.data != NULL) free_matrix(&layer->z);
    if (layer->a.data != NULL) free_matrix(&layer->a);
    if (layer->x.data != NULL) free_matrix(&layer->x);
}

// Forward pass through a layer
void layer_forward(Layer* layer, Matrix* input) {
    // Allocate if needed
    if (layer->z.data == NULL) {
        layer->z = create_matrix(layer->W.rows, input->cols);
        layer->a = create_matrix(layer->W.rows, input->cols);
        layer->x = create_matrix(input->rows, input->cols);
    }
    
    // Cache input
    matrix_copy(&layer->x, input);
    
    // z = W * x + b
    matrix_multiply(&layer->z, &layer->W, input);
    for (int i = 0; i < layer->z.rows; i++) {
        for (int j = 0; j < layer->z.cols; j++) {
            layer->z.data[i][j] += layer->b.data[i][0];
        }
    }
    
    // Apply activation
    apply_activation(&layer->a, &layer->z, layer->activation);
}

// Binary cross-entropy loss
double binary_cross_entropy(Matrix* y_true, Matrix* y_pred) {
    double loss = 0.0;
    int n = y_true->cols;
    double epsilon = 1e-15;
    
    for (int j = 0; j < n; j++) {
        double pred = y_pred->data[0][j];
        pred = (pred < epsilon) ? epsilon : pred;
        pred = (pred > 1 - epsilon) ? 1 - epsilon : pred;
        
        double true_val = y_true->data[0][j];
        loss += true_val * log(pred) + (1 - true_val) * log(1 - pred);
    }
    
    return -loss / n;
}

// Initialize MLP
void init_mlp(MLP* mlp, int* layer_sizes, ActivationType* activations, int num_layers) {
    mlp->num_layers = num_layers - 1;
    
    for (int i = 0; i < mlp->num_layers; i++) {
        init_layer(&mlp->layers[i], layer_sizes[i], layer_sizes[i+1], activations[i]);
    }
    
    mlp->max_losses = 10000;
    mlp->losses = (double*)malloc(mlp->max_losses * sizeof(double));
    mlp->num_losses = 0;
}

void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        free_layer(&mlp->layers[i]);
    }
    free(mlp->losses);
}

// Forward pass through MLP
void mlp_forward(MLP* mlp, Matrix* input, Matrix* output) {
    Layer* layer = &mlp->layers[0];
    layer_forward(layer, input);
    
    for (int i = 1; i < mlp->num_layers; i++) {
        layer_forward(&mlp->layers[i], &mlp->layers[i-1].a);
    }
    
    matrix_copy(output, &mlp->layers[mlp->num_layers - 1].a);
}

// Train MLP
void mlp_fit(MLP* mlp, double** X, double* y, int n_samples, int n_features, 
             double learning_rate, int epochs, int verbose) {
    // Transpose input
    Matrix X_T = create_matrix(n_features, n_samples);
    Matrix y_T = create_matrix(1, n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_T.data[j][i] = X[i][j];
        }
        y_T.data[0][i] = y[i];
    }
    
    Matrix y_pred = create_matrix(1, n_samples);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        mlp_forward(mlp, &X_T, &y_pred);
        
        // Compute loss
        double loss = binary_cross_entropy(&y_T, &y_pred);
        mlp->losses[mlp->num_losses++] = loss;
        
        // Backward pass and update (simplified for XOR)
        // delta = y_pred - y_true
        Matrix delta = create_matrix(1, n_samples);
        for (int j = 0; j < n_samples; j++) {
            delta.data[0][j] = y_pred.data[0][j] - y_T.data[0][j];
        }
        
        // Update output layer
        Layer* out_layer = &mlp->layers[mlp->num_layers - 1];
        Matrix x_T = create_matrix(out_layer->x.cols, out_layer->x.rows);
        matrix_transpose(&x_T, &out_layer->x);
        
        Matrix dW = create_matrix(out_layer->W.rows, out_layer->W.cols);
        matrix_multiply(&dW, &delta, &x_T);
        
        for (int i = 0; i < out_layer->W.rows; i++) {
            for (int j = 0; j < out_layer->W.cols; j++) {
                out_layer->W.data[i][j] -= learning_rate * dW.data[i][j] / n_samples;
            }
        }
        
        // Update bias
        for (int i = 0; i < out_layer->b.rows; i++) {
            double db = 0;
            for (int j = 0; j < n_samples; j++) {
                db += delta.data[i][j];
            }
            out_layer->b.data[i][0] -= learning_rate * db / n_samples;
        }
        
        // Propagate delta backwards
        Matrix W_T = create_matrix(out_layer->W.cols, out_layer->W.rows);
        matrix_transpose(&W_T, &out_layer->W);
        
        Matrix delta_prev = create_matrix(W_T.rows, delta.cols);
        matrix_multiply(&delta_prev, &W_T, &delta);
        
        // Update hidden layers
        for (int l = mlp->num_layers - 2; l >= 0; l--) {
            Layer* layer = &mlp->layers[l];
            
            // Apply activation derivative
            for (int i = 0; i < delta_prev.rows; i++) {
                for (int j = 0; j < delta_prev.cols; j++) {
                    if (layer->activation == TANH_ACTIVATION) {
                        delta_prev.data[i][j] *= tanh_derivative_func(layer->a.data[i][j]);
                    } else if (layer->activation == SIGMOID) {
                        delta_prev.data[i][j] *= sigmoid_derivative_func(layer->a.data[i][j]);
                    } else if (layer->activation == RELU) {
                        delta_prev.data[i][j] *= relu_derivative_func(layer->z.data[i][j]);
                    }
                }
            }
            
            // Update weights
            Matrix x_T_l = create_matrix(layer->x.cols, layer->x.rows);
            matrix_transpose(&x_T_l, &layer->x);
            
            Matrix dW_l = create_matrix(layer->W.rows, layer->W.cols);
            matrix_multiply(&dW_l, &delta_prev, &x_T_l);
            
            for (int i = 0; i < layer->W.rows; i++) {
                for (int j = 0; j < layer->W.cols; j++) {
                    layer->W.data[i][j] -= learning_rate * dW_l.data[i][j] / n_samples;
                }
            }
            
            // Update bias
            for (int i = 0; i < layer->b.rows; i++) {
                double db = 0;
                for (int j = 0; j < n_samples; j++) {
                    db += delta_prev.data[i][j];
                }
                layer->b.data[i][0] -= learning_rate * db / n_samples;
            }
            
            if (l > 0) {
                Matrix W_T_l = create_matrix(layer->W.cols, layer->W.rows);
                matrix_transpose(&W_T_l, &layer->W);
                
                Matrix new_delta = create_matrix(W_T_l.rows, delta_prev.cols);
                matrix_multiply(&new_delta, &W_T_l, &delta_prev);
                
                free_matrix(&delta_prev);
                delta_prev = new_delta;
                free_matrix(&W_T_l);
            }
            
            free_matrix(&x_T_l);
            free_matrix(&dW_l);
        }
        
        free_matrix(&delta);
        free_matrix(&x_T);
        free_matrix(&dW);
        free_matrix(&W_T);
        free_matrix(&delta_prev);
        
        if (verbose && (epoch + 1) % 1000 == 0) {
            printf("Epoch %d/%d, Loss: %.6f\n", epoch + 1, epochs, loss);
        }
    }
    
    if (verbose) {
        printf("\nTraining completed. Final loss: %.6f\n", mlp->losses[mlp->num_losses - 1]);
    }
    
    free_matrix(&X_T);
    free_matrix(&y_T);
    free_matrix(&y_pred);
}

// Predict
void mlp_predict(MLP* mlp, double** X, int n_samples, int n_features, int* predictions) {
    Matrix X_T = create_matrix(n_features, n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_T.data[j][i] = X[i][j];
        }
    }
    
    Matrix y_pred = create_matrix(1, n_samples);
    mlp_forward(mlp, &X_T, &y_pred);
    
    for (int i = 0; i < n_samples; i++) {
        predictions[i] = (y_pred.data[0][i] >= 0.5) ? 1 : 0;
    }
    
    free_matrix(&X_T);
    free_matrix(&y_pred);
}

int main() {
    printf("Training MLP on XOR problem\n\n");
    
    // XOR dataset
    double X_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double y_data[4] = {0, 1, 1, 0};
    
    double** X = (double**)malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) {
        X[i] = X_data[i];
    }
    
    // Create MLP
    int layer_sizes[] = {2, 4, 1};
    ActivationType activations[] = {TANH_ACTIVATION, SIGMOID};
    
    MLP mlp;
    init_mlp(&mlp, layer_sizes, activations, 3);
    
    // Train
    mlp_fit(&mlp, X, y_data, 4, 2, 0.5, 5000, 1);
    
    // Test predictions
    printf("\n--- Predictions ---\n");
    int predictions[4];
    mlp_predict(&mlp, X, 4, 2, predictions);
    
    for (int i = 0; i < 4; i++) {
        printf("Input: [%.0f, %.0f], Predicted: %d, Actual: %.0f\n", 
               X[i][0], X[i][1], predictions[i], y_data[i]);
    }
    
    // Cleanup
    free_mlp(&mlp);
    free(X);
    
    return 0;
}
