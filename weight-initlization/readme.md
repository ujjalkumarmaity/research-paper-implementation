Weight Initialization is crucial step for convergence model. There are several way we can initialize weights. 

## Why we don't use "Zero/Constant” Initialization?

If we initialize all weights to zero, every neuron in a hidden layer will perform the same calculation and produce the same output. During backpropagation, they will all receive the identical gradient update.

- **The Result:** The neurons remain "symmetrical." They never learn different features, effectively turning a deep network into a single-neuron model. This is called the **Symmetry Breaking** problem.

### what is symmetry problem?

The **Symmetry Problem** occurs when weights in a neural network are initialized to the same value. If multiple neurons in a layer have the same weights and biases, they will produce the same output and receive the same gradient updates, effectively acting as a single neuron. This prevents the network from learning complex, diverse features.

1. Forward Pass Symmetry -

for hidden layer $l$, pre activation output - 

$z_j^{[l]} = \sum_{i} w_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]}  \quad z_k^{[l]} = \sum_{i} w_{ki}^{[l]} a_i^{[l-1]} + b_k^{[l]}$

If we initialize $w_{ji}^{[l]} = w_{ki}^{[l]}$ and $b_j^{[l]} = b_k^{[l]}$, then for any input: $z_j^{[l]} = z_k^{[l]} \implies a_j^{[l]} = a_k^{[l]}$

So these two neuron produce identical output. 

1. During backpropagation 

$\delta_j^{[l]} = \left( \sum_{m} \delta_m^{[l+1]} w_{mj}^{[l+1]} \right) \sigma'(z_j^{[l]})$

if weights across layer same then $\delta_j^{[l]} = \delta_k^{[l]}$

1. Identical Weight Updates

Since $\delta_j^{[l]} = \delta_k^{[l]}$ and the inputs $a_i^{[l-1]}$ are the same for both, the gradients are identical: $\frac{\partial L}{\partial w_{ji}^{[l]}} = \frac{\partial L}{\partial w_{ki}^{[l]}}$

$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$

because gradients are identical, weights will remain identical after every iteration.

**How to Break Symmetry?**

To break symmetry, we use random(Xavier or He) weight initialization.  

## 1. Xavier (Glorot) Initialization

“The goal of Xavier Initialization is to initialize the weights such that the **variance of the activations are the same across every layer**. This constant variance helps prevent the gradient from exploding or vanishing.”

Designed for layers using Sigmoid or Tanh activation functions. It keeps the variance of the input and output the same.

### Mathematical Intuition

- Activations are **zero-mean** (**This assumption breaks for ReLU**, which is not zero-mean)
- Weights and inputs are centered at zero
- Weights and inputs are independent and identically distributed
- **Activation function is approximately linear near 0**

So here goal is keep variance same across layer 

**1. Assumption of Constant Variance:**

$Var(a_i^{[\ell-1]}) = Var(a_i^{[\ell]})$

**2. Linearity Approximation:**

$= Var(z_i^{[\ell]})$

> Side note: linearity of tanh around zero, $tanh(z) \approx z$
> 

**3. Expansion of $z$:**

$= Var\left( \sum_{j=1}^{n^{[\ell-1]}} w_{ij}^{[\ell]} a_j^{[\ell-1]} \right)$

**4. Summation Variance:**

$= \sum_{j=1}^{n^{[\ell-1]}} Var(w_{ij}^{[\ell]} a_j^{[\ell-1]})$

> Side note: variance of independent sum, $Var(X+Y) = Var(X) + Var(Y)$
> 

**5. Product Variance Expansion:**

$= \sum_{j=1}^{n^{[\ell-1]}} \left( E[w_{ij}^{[\ell]}]^2 Var(a_j^{[\ell-1]}) + E[a_j^{[\ell-1]}]^2 Var(w_{ij}^{[\ell]}) + Var(w_{ij}^{[\ell]})Var(a_j^{[\ell-1]}) \right)$

> Side note: variance of independent product, $Var(XY) = E[X]^2Var(Y) + E[Y]^2Var(X) + Var(X)Var(Y)$
> 

**6. Final Simplification & Result:**

we assume activations and variance are **zero-mean.** 

so $E[w_{ij}^{[\ell]}] = 0$ and $E[a_j^{[\ell-1]}] = 0$

$$
⁍
$$

So $\text{Var}(W) = \frac{1}{n_{in} }$ equation control only forward pass.

$\text{Var}(W) = \frac{1}{n_{out} }$ equation control only forward pass.

By taking **averaging of two constraints**:

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

### He (Kaiming) Initialization

Designed specifically for ReLU (Rectified Linear Unit) activation functions. Since ReLU "shuts off" half the neurons, the weights need to be slightly larger to compensate.

Weights are picked from a distribution with a variance of:

$$
\text{Var}(W) = \frac{2}{n_{in}}
$$

---

## Referance

- https://cs230.stanford.edu/section/4/
- https://www.pinecone.io/learn/weight-initialization/
