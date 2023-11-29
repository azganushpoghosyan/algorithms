"""
The perceptron is a basic supervised learning algorithm for binary classification tasks.
It consists of a single-layer neural network with adjustable weights and a threshold.
The model takes input features, multiplies them by corresponding weights, sums the results,
and applies a step function based on a threshold to make a binary prediction.
Training involves adjusting the weights based on misclassifications,
aiming to find values that correctly separate the classes.
While limited to linearly separable problems, the perceptron serves as the foundation
for more complex neural network architectures.
"""
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # Initialize weights to zeros, including the bias term
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Step function as the activation function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Calculate the weighted sum and apply the activation function
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        # Training the perceptron
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                # Make a prediction
                prediction = self.predict(inputs)

                # Update weights based on prediction error
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)