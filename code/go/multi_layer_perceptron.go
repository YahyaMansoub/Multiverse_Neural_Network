/*
 * Multi-Layer Perceptron Implementation in Go
 * A neural network with multiple hidden layers for binary classification.
 */

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ActivationType represents different activation functions
type ActivationType int

const (
	Sigmoid ActivationType = iota
	ReLU
	Tanh
)

// Matrix represents a 2D matrix
type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{Rows: rows, Cols: cols, Data: data}
}

// Multiply performs matrix multiplication
func (m *Matrix) Multiply(other *Matrix) *Matrix {
	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			for k := 0; k < m.Cols; k++ {
				result.Data[i][j] += m.Data[i][k] * other.Data[k][j]
			}
		}
	}
	return result
}

// Add performs element-wise addition
func (m *Matrix) Add(other *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}
	return result
}

// Subtract performs element-wise subtraction
func (m *Matrix) Subtract(other *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] - other.Data[i][j]
		}
	}
	return result
}

// Transpose returns the transpose of the matrix
func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

// Apply applies a function element-wise
func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// ElementwiseMultiply performs element-wise multiplication
func (m *Matrix) ElementwiseMultiply(other *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * other.Data[i][j]
		}
	}
	return result
}

// Copy creates a copy of the matrix
func (m *Matrix) Copy() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		copy(result.Data[i], m.Data[i])
	}
	return result
}

// Activation functions
func sigmoid(z float64) float64 {
	if z < -500 {
		z = -500
	}
	if z > 500 {
		z = 500
	}
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidDerivative(a float64) float64 {
	return a * (1.0 - a)
}

func relu(z float64) float64 {
	return math.Max(0, z)
}

func reluDerivative(z float64) float64 {
	if z > 0 {
		return 1.0
	}
	return 0.0
}

func tanhFunc(z float64) float64 {
	return math.Tanh(z)
}

func tanhDerivative(a float64) float64 {
	return 1.0 - a*a
}

// Layer represents a single fully-connected layer
type Layer struct {
	W          *Matrix
	B          *Matrix
	Z          *Matrix
	A          *Matrix
	X          *Matrix
	Activation ActivationType
}

// NewLayer creates a new layer with given dimensions
func NewLayer(inputSize, outputSize int, activation ActivationType) *Layer {
	layer := &Layer{
		W:          NewMatrix(outputSize, inputSize),
		B:          NewMatrix(outputSize, 1),
		Activation: activation,
	}

	// Xavier/Glorot initialization
	limit := math.Sqrt(6.0 / float64(inputSize+outputSize))
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			layer.W.Data[i][j] = rand.Float64()*2*limit - limit
		}
	}

	return layer
}

// Forward performs forward pass through the layer
func (l *Layer) Forward(input *Matrix) *Matrix {
	l.X = input.Copy()

	// z = W * x + b
	l.Z = l.W.Multiply(input)
	for i := 0; i < l.Z.Rows; i++ {
		for j := 0; j < l.Z.Cols; j++ {
			l.Z.Data[i][j] += l.B.Data[i][0]
		}
	}

	// Apply activation
	switch l.Activation {
	case Sigmoid:
		l.A = l.Z.Apply(sigmoid)
	case ReLU:
		l.A = l.Z.Apply(relu)
	case Tanh:
		l.A = l.Z.Apply(tanhFunc)
	}

	return l.A
}

// Backward performs backward pass through the layer
func (l *Layer) Backward(deltaNext *Matrix) (*Matrix, *Matrix, *Matrix) {
	batchSize := float64(l.X.Cols)

	// Apply activation derivative
	var delta *Matrix
	if l.Activation == Sigmoid || l.Activation == Tanh {
		var derivFn func(float64) float64
		if l.Activation == Sigmoid {
			derivFn = sigmoidDerivative
		} else {
			derivFn = tanhDerivative
		}
		deriv := l.A.Apply(derivFn)
		delta = deltaNext.ElementwiseMultiply(deriv)
	} else { // ReLU
		deriv := l.Z.Apply(reluDerivative)
		delta = deltaNext.ElementwiseMultiply(deriv)
	}

	// Compute gradients
	dW := delta.Multiply(l.X.Transpose())
	for i := 0; i < dW.Rows; i++ {
		for j := 0; j < dW.Cols; j++ {
			dW.Data[i][j] /= batchSize
		}
	}

	dB := NewMatrix(delta.Rows, 1)
	for i := 0; i < delta.Rows; i++ {
		sum := 0.0
		for j := 0; j < delta.Cols; j++ {
			sum += delta.Data[i][j]
		}
		dB.Data[i][0] = sum / batchSize
	}

	// Gradient for previous layer
	deltaPrev := l.W.Transpose().Multiply(delta)

	return deltaPrev, dW, dB
}

// MLP represents a multi-layer perceptron
type MLP struct {
	Layers []Layer
	Losses []float64
}

// NewMLP creates a new MLP with given architecture
func NewMLP(layerSizes []int, activations []ActivationType) *MLP {
	mlp := &MLP{
		Layers: make([]Layer, len(layerSizes)-1),
		Losses: make([]float64, 0),
	}

	for i := 0; i < len(layerSizes)-1; i++ {
		mlp.Layers[i] = *NewLayer(layerSizes[i], layerSizes[i+1], activations[i])
	}

	return mlp
}

// Forward performs forward pass through the MLP
func (mlp *MLP) Forward(X *Matrix) *Matrix {
	a := X
	for i := range mlp.Layers {
		a = mlp.Layers[i].Forward(a)
	}
	return a
}

// BinaryCrossEntropy computes binary cross-entropy loss
func BinaryCrossEntropy(yTrue, yPred *Matrix) float64 {
	loss := 0.0
	epsilon := 1e-15
	n := float64(yTrue.Cols)

	for j := 0; j < yTrue.Cols; j++ {
		pred := yPred.Data[0][j]
		pred = math.Max(epsilon, math.Min(1.0-epsilon, pred))
		trueVal := yTrue.Data[0][j]
		loss += trueVal*math.Log(pred) + (1.0-trueVal)*math.Log(1.0-pred)
	}

	return -loss / n
}

// Backward performs backward pass through the MLP
func (mlp *MLP) Backward(yTrue, yPred *Matrix) [][2]*Matrix {
	// Initial gradient
	delta := yPred.Subtract(yTrue)

	gradients := make([][2]*Matrix, len(mlp.Layers))

	// Backpropagate
	for i := len(mlp.Layers) - 1; i >= 0; i-- {
		var deltaPrev, dW, dB *Matrix

		if i == len(mlp.Layers)-1 {
			// Output layer
			batchSize := float64(mlp.Layers[i].X.Cols)
			dW = delta.Multiply(mlp.Layers[i].X.Transpose())
			for r := 0; r < dW.Rows; r++ {
				for c := 0; c < dW.Cols; c++ {
					dW.Data[r][c] /= batchSize
				}
			}

			dB = NewMatrix(delta.Rows, 1)
			for r := 0; r < delta.Rows; r++ {
				sum := 0.0
				for c := 0; c < delta.Cols; c++ {
					sum += delta.Data[r][c]
				}
				dB.Data[r][0] = sum / batchSize
			}

			deltaPrev = mlp.Layers[i].W.Transpose().Multiply(delta)
		} else {
			// Hidden layers
			deltaPrev, dW, dB = mlp.Layers[i].Backward(delta)
		}

		gradients[i] = [2]*Matrix{dW, dB}
		delta = deltaPrev
	}

	return gradients
}

// UpdateParameters updates layer parameters using gradients
func (mlp *MLP) UpdateParameters(gradients [][2]*Matrix, learningRate float64) {
	for i := range mlp.Layers {
		dW := gradients[i][0]
		dB := gradients[i][1]

		for r := 0; r < mlp.Layers[i].W.Rows; r++ {
			for c := 0; c < mlp.Layers[i].W.Cols; c++ {
				mlp.Layers[i].W.Data[r][c] -= learningRate * dW.Data[r][c]
			}
		}

		for r := 0; r < mlp.Layers[i].B.Rows; r++ {
			mlp.Layers[i].B.Data[r][0] -= learningRate * dB.Data[r][0]
		}
	}
}

// Fit trains the MLP
func (mlp *MLP) Fit(X *Matrix, y []float64, learningRate float64, epochs int, verbose bool) {
	// Transpose input
	XT := X.Transpose()
	yT := NewMatrix(1, len(y))
	for i := range y {
		yT.Data[0][i] = y[i]
	}

	mlp.Losses = make([]float64, 0)

	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		yPred := mlp.Forward(XT)

		// Compute loss
		loss := BinaryCrossEntropy(yT, yPred)
		mlp.Losses = append(mlp.Losses, loss)

		// Backward pass
		gradients := mlp.Backward(yT, yPred)

		// Update parameters
		mlp.UpdateParameters(gradients, learningRate)

		// Print progress
		if verbose && (epoch+1)%1000 == 0 {
			fmt.Printf("Epoch %d/%d, Loss: %.6f\n", epoch+1, epochs, loss)
		}
	}

	if verbose {
		fmt.Printf("\nTraining completed. Final loss: %.6f\n", mlp.Losses[len(mlp.Losses)-1])
	}
}

// PredictProba predicts probabilities
func (mlp *MLP) PredictProba(X *Matrix) *Matrix {
	XT := X.Transpose()
	result := mlp.Forward(XT)
	return result.Transpose()
}

// Predict predicts class labels
func (mlp *MLP) Predict(X *Matrix, threshold float64) []int {
	proba := mlp.PredictProba(X)
	predictions := make([]int, proba.Rows)

	for i := 0; i < proba.Rows; i++ {
		if proba.Data[i][0] >= threshold {
			predictions[i] = 1
		} else {
			predictions[i] = 0
		}
	}

	return predictions
}

// Score computes accuracy
func (mlp *MLP) Score(X *Matrix, y []float64) float64 {
	predictions := mlp.Predict(X, 0.5)
	correct := 0

	for i := range y {
		if predictions[i] == int(y[i]) {
			correct++
		}
	}

	return float64(correct) / float64(len(y))
}

func main() {
	fmt.Println("Training MLP on XOR problem\n")

	// XOR dataset
	X := NewMatrix(4, 2)
	X.Data = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	y := []float64{0, 1, 1, 0}

	// Create and train MLP
	mlp := NewMLP([]int{2, 4, 1}, []ActivationType{Tanh, Sigmoid})
	mlp.Fit(X, y, 0.5, 5000, true)

	// Test predictions
	fmt.Println("\n--- Predictions ---")
	predictions := mlp.Predict(X, 0.5)
	probas := mlp.PredictProba(X)

	for i := 0; i < 4; i++ {
		fmt.Printf("Input: [%.0f, %.0f], Predicted: %d, Probability: %.4f, Actual: %.0f\n",
			X.Data[i][0], X.Data[i][1], predictions[i], probas.Data[i][0], y[i])
	}

	fmt.Printf("\nAccuracy: %.0f%%\n", mlp.Score(X, y)*100)
}
