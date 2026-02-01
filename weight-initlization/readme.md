Weight Initilization is crutial step for convergece model. There are several way we can intilize weights. Let's discuss few of them-

1. Zero Initialization:
   - All weights are initialized to zero.
   - Problem: This causes neurons to learn the same features during training, leading to poor performance.

2. Random Initialization:
   - Weights are initialized randomly, often using a uniform or normal distribution.
   - Helps break symmetry, allowing different neurons to learn different features.

3. Xavier/Glorot Initialization:
   - Designed for layers with sigmoid or tanh activation functions.
   - Weights are initialized from a distribution with zero mean and a specific variance to maintain the variance of activations through layers.

4. He Initialization:
   - Designed for layers with ReLU activation functions.
   - Weights are initialized from a distribution with zero mean and a variance of 2/n, where n is the number of input units in the layer.
