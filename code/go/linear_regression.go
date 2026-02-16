/**
 * Linear Regression from First Principles - Go Implementation
 * Implements gradient descent-based linear regression with MSE loss.
 */

package main

import (
	"fmt"
	"math"
	"math/rand"
)

// LinearRegression represents a linear regression model
type LinearRegression struct {
	W         []float64   // Weight vector
	B         float64     // Bias term
	Losses    []float64   // Training loss history
	NFeatures int         // Number of features
}

// NewLinearRegression creates a new linear regression model
func NewLinearRegression() *LinearRegression {
	return &LinearRegression{
		Losses: make([]float64, 0),
	}
}

// forward computes predictions: y_hat = X @ w + b
func (lr *LinearRegression) forward(X [][]float64) []float64 {
	nSamples := len(X)
	yHat := make([]float64, nSamples)
	
	for i := 0; i < nSamples; i++ {
		yHat[i] = lr.B
		for j := 0; j < lr.NFeatures; j++ {
			yHat[i] += X[i][j] * lr.W[j]
		}
	}
	
	return yHat
}

// computeLoss computes MSE loss
func (lr *LinearRegression) computeLoss(yHat, y []float64) float64 {
	sum := 0.0
	nSamples := len(y)
	
	for i := 0; i < nSamples; i++ {
		diff := yHat[i] - y[i]
		sum += diff * diff
	}
	
	return sum / float64(nSamples)
}

// computeGradients computes gradients of loss w.r.t. w and b
func (lr *LinearRegression) computeGradients(X [][]float64, y, yHat []float64) ([]float64, float64) {
	nSamples := len(X)
	dw := make([]float64, lr.NFeatures)
	db := 0.0
	
	// Compute gradients
	for i := 0; i < nSamples; i++ {
		error := yHat[i] - y[i]
		db += error
		for j := 0; j < lr.NFeatures; j++ {
			dw[j] += error * X[i][j]
		}
	}
	
	// Scale by 2/n
	scale := 2.0 / float64(nSamples)
	db *= scale
	for j := 0; j < lr.NFeatures; j++ {
		dw[j] *= scale
	}
	
	return dw, db
}

// Fit trains the model using gradient descent
func (lr *LinearRegression) Fit(X [][]float64, y []float64, learningRate float64, epochs int, verbose bool) {
	nSamples := len(X)
	lr.NFeatures = len(X[0])
	
	// Initialize weights randomly
	rand.Seed(42)
	lr.W = make([]float64, lr.NFeatures)
	for j := 0; j < lr.NFeatures; j++ {
		lr.W[j] = rand.NormFloat64()
	}
	lr.B = 0.0
	
	// Training loop
	lr.Losses = make([]float64, 0, epochs)
	
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		yHat := lr.forward(X)
		
		// Compute loss
		loss := lr.computeLoss(yHat, y)
		lr.Losses = append(lr.Losses, loss)
		
		// Compute gradients
		dw, db := lr.computeGradients(X, y, yHat)
		
		// Update parameters
		for j := 0; j < lr.NFeatures; j++ {
			lr.W[j] -= learningRate * dw[j]
		}
		lr.B -= learningRate * db
		
		// Print progress
		if verbose && (epoch+1)%200 == 0 {
			fmt.Printf("Epoch %d/%d, Loss: %.6f\n", epoch+1, epochs, loss)
		}
	}
	
	if verbose {
		fmt.Printf("\nTraining completed. Final loss: %.6f\n", lr.Losses[len(lr.Losses)-1])
	}
}

// Predict makes predictions on new data
func (lr *LinearRegression) Predict(X [][]float64) []float64 {
	if lr.W == nil {
		panic("Model must be fitted before making predictions")
	}
	return lr.forward(X)
}

// GetParameters returns the learned parameters
func (lr *LinearRegression) GetParameters() ([]float64, float64) {
	return lr.W, lr.B
}

// Score computes R² score
func (lr *LinearRegression) Score(X [][]float64, y []float64) float64 {
	yHat := lr.Predict(X)
	
	// Compute mean of y
	yMean := 0.0
	for _, val := range y {
		yMean += val
	}
	yMean /= float64(len(y))
	
	// Compute SS_res and SS_tot
	ssRes := 0.0
	ssTot := 0.0
	for i := 0; i < len(y); i++ {
		diffRes := y[i] - yHat[i]
		diffTot := y[i] - yMean
		ssRes += diffRes * diffRes
		ssTot += diffTot * diffTot
	}
	
	return 1.0 - (ssRes / ssTot)
}

// Helper function to generate random normal data
func randn() float64 {
	return rand.NormFloat64()
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("Linear Regression from First Principles - Go Implementation")
	fmt.Println("============================================================")
	
	// Set random seed
	rand.Seed(42)
	
	// Generate synthetic data
	nSamples := 100
	nFeatures := 2
	
	trueW := []float64{3.0, 2.0}
	trueB := 1.0
	
	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)
	
	for i := 0; i < nSamples; i++ {
		X[i] = make([]float64, nFeatures)
		sum := trueB
		for j := 0; j < nFeatures; j++ {
			X[i][j] = randn()
			sum += X[i][j] * trueW[j]
		}
		y[i] = sum + 0.5*randn()
	}
	
	fmt.Printf("\nData shape: X=(%d, %d), y=(%d)\n", nSamples, nFeatures, nSamples)
	fmt.Printf("True parameters: w=[%.1f, %.1f], b=%.4f\n", trueW[0], trueW[1], trueB)
	
	// Create and train model
	fmt.Println("\n------------------------------------------------------------")
	fmt.Println("Training Model...")
	fmt.Println("------------------------------------------------------------")
	
	model := NewLinearRegression()
	model.Fit(X, y, 0.1, 1000, true)
	
	// Get learned parameters
	wLearned, bLearned := model.GetParameters()
	fmt.Printf("\nLearned parameters: w=[%.5f, %.5f], b=%.4f\n", wLearned[0], wLearned[1], bLearned)
	fmt.Printf("True parameters: w=[%.1f, %.1f], b=%.4f\n", trueW[0], trueW[1], trueB)
	
	// Compute R² score
	r2 := model.Score(X, y)
	fmt.Printf("\nR² Score: %.6f\n", r2)
	
	// Make predictions on new data
	fmt.Println("\n------------------------------------------------------------")
	fmt.Println("Making Predictions on New Data...")
	fmt.Println("------------------------------------------------------------")
	
	XNew := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{0.5, 1.5},
	}
	
	predictions := model.Predict(XNew)
	
	fmt.Println("\nNew data:")
	for _, row := range XNew {
		fmt.Printf("[%.1f, %.1f]\n", row[0], row[1])
	}
	
	fmt.Println("\nPredictions:")
	for _, pred := range predictions {
		fmt.Printf("%.5f\n", pred)
	}
	
	fmt.Println("\n============================================================")
	fmt.Println("Example completed successfully!")
	fmt.Println("============================================================")
}
