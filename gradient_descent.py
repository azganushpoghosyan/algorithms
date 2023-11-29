"""
Intro to Gradient Descent algorithm
Gradient Descent is a fundamental optimization algorithm in machine learning,
specifically designed to minimize a cost or loss function iteratively.
At its core, the algorithm adjusts model parameters by computing the gradient of the cost function,
representing the direction of the steepest ascent.
The learning rate, a hyperparameter, controls the step size during each iteration.
The process involves:
    1) initializing parameters,
    2) computing the gradient,
    3) updating parameters in the opposite direction of the gradient,
    4) repeating until convergence.
Various forms of Gradient Descent exist, including Batch Gradient Descent, which uses the entire dataset,
Stochastic Gradient Descent (SGD) employing a single data point per iteration,
and Mini-Batch Gradient Descent, striking a balance with random subsets.
The algorithm's success lies in its versatility, forming the foundation for training diverse
machine learning models such as linear regression and neural networks.
"""
import numpy as np

def loss_fn(beta, X, y):
    """Calculate the loss function (MSE)."""
    return np.sum(np.square(y - X.dot(beta)))

def loss_grad(beta, X, y):
    """Calculate the gradient of the loss function."""
    return -2 * X.T.dot(y - X.dot(beta))

def gradient_step(beta, step_size, X, y):
    """Perform a gradient descent step."""
    loss, grads = loss_fn(beta, X, y), loss_grad(beta, X, y)
    beta = beta - step_size * grads
    return loss, beta

def gradient_descent(X, y, step_size, precision, max_iter=10000, warn_max_iter=True):
    """Perform gradient descent optimization."""
    beta = np.zeros((X.shape[1], 1))  # Initialize beta with zeros
    beta_last = beta  # Initialize beta_last outside the loop
    losses = []  # Array for recording the value of the loss over the iterations.
    beta_steps = []  # Array for recording the value of beta at each iteration
    graceful = False

    for _ in range(max_iter):
        beta_last = beta
        loss, beta = gradient_step(beta, step_size, X, y)
        losses.append(loss)
        beta_steps.append(beta.copy())

        # Use the norm of the difference between the new beta and the old beta as a stopping criterion
        if np.linalg.norm((beta - beta_last) / beta) < precision:
            graceful = True
            break

    if not graceful and warn_max_iter:
        print("Reached max iterations.")

    return beta, np.array(losses), np.array(beta_steps)