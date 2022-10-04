# Overview

In this repository, I implement a mini version of the linear regression model using gradient descent to arrive at the values of the weights that'd give the best fit line for any given dataset.

Model Implementation: [linear_regression_from_scratch](./MyLinearRegression/linear_regression_from_scratch.py)

Trying out the model on custom dataset: [custom_model_in_action](./custom_model_in_action.ipynb)

For the custom dataset that I created and used to test this model, a correct implementation of the model should give the weights (theta) as

$$\theta = \left(\begin{array}{cc} 1.5\\\\2\\\\1\end{array}\right)$$

# The Hypothesis - $h_{\theta}(x^{(i)})$

Instead of taking each feature, $x_{n}^{(i)}$, one at a time in a training example to multiply with its corresponding $\theta_{j}$, a more conservative way to do this is to vectorize the $\theta$. Making $\theta$ an $(n+1)$ dimensional vector, with the first element being the constant term in the expression $h_{\theta}(x^{(i)}) = \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} + ... + \theta_{n}x_{n}^{(i)}$ and the rest of the elements being the coefficients of $x_{1}^{(i)}, x_{2}^{(i)}, ..., x_{n}^{(i)}$ where $n$ is the number of features of each training example. Multiplying the design matrix, X (an $m\times(n+1)$ matrix), by $\theta$ will produce an $m$ dimensional vector (an $m\times1$ matrix). This vector contains the predictions of the hypothesis.

$$h_{\theta}(x^{(i)})=X\theta$$

# The Cost Function for Linear Regression

$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^{2}$$

# The Gradient Descent

$$\theta_{j}:=\theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta)$$

But since I have decided to vectorize the $\theta$, I can think of the gradient descent as:

$$\theta:=\theta-\alpha\delta$$

where $\delta$ will be an $(n+1)$ dimensional vector given by:

$$\delta=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}$$
