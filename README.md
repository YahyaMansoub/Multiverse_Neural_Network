# Multiverse Neural Network

A multi-language neural network project exploring traditional and custom architectures across different programming paradigms. This project implements neural networks from scratch in multiple languages, allowing for comparative analysis of computational performance and language-specific approaches.

## 🎯 Project Goals

- Implement neural networks from scratch across multiple programming languages
- Compare computational performance between different implementations
- Explore cross-language hybrid approaches in the MELANGE directory
- Build a comprehensive understanding of neural networks at the algorithmic level

## 📁 Project Structure

```
Multiverse_Neural_Network/
├── notebooks/              # Jupyter notebooks with explanations & experiments
│   ├── 1_linear_regression.ipynb
│   └── 2_multi_layer_perceptron.ipynb
│
├── code/                   # Language-specific implementations
│   ├── python/
│   │   ├── linear_regression.py
│   │   └── multi_layer_perceptron.py
│   ├── c/
│   │   ├── linear_regression.c
│   │   └── multi_layer_perceptron.c
│   ├── cpp/
│   │   ├── linear_regression.cpp
│   │   └── multi_layer_perceptron.cpp
│   ├── go/
│   │   ├── linear_regression.go
│   │   └── multi_layer_perceptron.go
│   └── java/
│       ├── LinearRegression.java
│       └── MultiLayerPerceptron.java
│
└── MELANGE/                # Cross-language experiments
    └── README.md
```

## 🧠 Implementations

### Current Models

#### 1. Linear Regression
Basic supervised learning model for regression tasks.
- **Languages**: Python, C, C++, Go, Java
- **Features**: Gradient descent, mean squared error loss

#### 2. Multi-Layer Perceptron (MLP)
Feedforward neural network for binary classification.
- **Languages**: Python, C, C++, Go, Java
- **Features**:
  - Multiple hidden layers
  - Activation functions: Sigmoid, ReLU, Tanh
  - Xavier/Glorot weight initialization
  - Backpropagation with gradient descent
  - Binary cross-entropy loss
- **Example Problem**: XOR classification

### Features by Language

| Language | Matrix Ops | OOP | Memory Management | Performance |
|----------|-----------|-----|-------------------|-------------|
| Python   | NumPy     | ✓   | Automatic         | Baseline    |
| C        | Manual    | ✗   | Manual            | Fastest     |
| C++      | STL       | ✓   | RAII              | Very Fast   |
| Go       | Slices    | ✓   | GC                | Fast        |
| Java     | Arrays    | ✓   | GC                | Fast        |

## 🚀 Running the Code

### Python
```bash
cd code/python
python multi_layer_perceptron.py
```

### C
```bash
cd code/c
gcc -o mlp multi_layer_perceptron.c -lm
./mlp
```

### C++
```bash
cd code/cpp
g++ -o mlp multi_layer_perceptron.cpp -std=c++17
./mlp
```

### Go
```bash
cd code/go
go run multi_layer_perceptron.go
```

### Java
```bash
cd code/java
javac MultiLayerPerceptron.java
java MultiLayerPerceptron
```

## 🧪 MELANGE: Cross-Language Experiments

The MELANGE directory is dedicated to experimental cross-language neural network implementations. This is where we explore:

- FFI (Foreign Function Interface) between languages
- Hybrid implementations combining strengths of different languages
- Performance optimization through polyglot approaches
- Unconventional architectures that leverage language-specific features

See [MELANGE/README.md](MELANGE/README.md) for more details.

## 📊 Future Work

- [ ] Convolutional Neural Networks (CNNs)
- [ ] Recurrent Neural Networks (RNNs)
- [ ] Transformer architecture
- [ ] Performance benchmarking across languages
- [ ] GPU acceleration implementations
- [ ] Cross-language hybrid models in MELANGE

## 🛠️ Technical Details

### MLP Architecture

The Multi-Layer Perceptron implementation includes:

1. **Layer Class**: Fully-connected layer with forward/backward propagation
2. **Activation Functions**: Sigmoid, ReLU, Tanh with derivatives
3. **Loss Function**: Binary cross-entropy for classification
4. **Optimizer**: Stochastic gradient descent
5. **Weight Initialization**: Xavier/Glorot initialization

### Algorithm Flow

```
1. Initialize weights randomly (Xavier initialization)
2. For each epoch:
   a. Forward pass: compute predictions
   b. Compute loss
   c. Backward pass: compute gradients
   d. Update weights using gradient descent
3. Return trained model
```

## 📚 Resources

- Notebooks contain detailed explanations and visualizations
- Each implementation includes comments explaining the algorithm
- XOR problem used as standard test case for non-linear classification

## 🤝 Contributing

This is a personal learning project, but suggestions and improvements are welcome!

## 📄 License

This project is open source and available for educational purposes.
