# BayGW

## Under construction 🛠️

Warning! This project is in an active-development phase. 

## Overview of the mathematics

The normalizing flow is a series of invertible maps which transform a D-dimensional standard normal variable $z$ to our target distribution,

$$x = (f_{K} \circ f_{K-1} \circ \dots \circ f_{1})(z)$$

To compute the log-likelihood of $x$, we can use the Jacobian transformation

$$\log p_{X}(x) = \log p_{Z}(z) + \sum_{i=1}^{K} \log \left| \mathrm{det} \frac{\partial f_i(z)}{\partial z} \right|$$

The latent distribution is a simple Gaussian with log probability $\log p_{Z}(z) = -\tfrac{1}{2} (z^T z + D\log(2\pi))$. The negative log likelihood (NLL) for a single data point is therefore

$$\mathrm{NLL}(x) = -\log p_{X}(x) = \frac{1}{2} (z^T z + D\log(2\pi)) + \sum_{i=1}^{K} \log \left| \mathrm{det} \frac{\partial f_i(z)}{\partial z} \right|$$
