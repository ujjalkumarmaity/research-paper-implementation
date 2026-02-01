import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List

class NNModel(object):
    def __init__(self, init_method: str, layers: List[int]):
        self.init_method = init_method
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = []
        self.gradient = []

    def _initilize_weights(self):

        for i in range(len(self.layers) - 1):
            f_in = self.layers[i]
            f_out = self.layers[i + 1]
            if self.init_method == "zero":
                w = np.zeros((f_in, f_out), dtype=float)
            elif self.init_method == "ones":
                w = np.ones((f_in, f_out), dtype=float)
            elif self.init_method == "random":
                w = np.random.randn(f_in, f_out) * 0.01
            elif self.init_method == "normalize":
                w = np.random.randn(f_in, f_out) / np.sqrt(f_in)
            elif self.init_method == "xavier": # good for tanh and sigmoid activation function 
                limit = np.sqrt(6 / (f_in + f_out))
                w = np.random.uniform(-limit, limit, (f_in, f_out))

            elif self.init_method == "he": # use for relu activation function
                w = np.random.randn(f_in, f_out) * np.sqrt(2 / f_in)

            else:
                raise ValueError(
                    f"Unknown weight initilization method - {self.init_method}"
                )
            b = np.zeros((1, f_out))
            self.biases.append(b)
            self.weights.append(w)

    def forward(self, x):
        self.activations = [x]
        last_act = x
        for i in range(len(self.weights)):
            # print(f"last_act shape - {last_act.shape}, self.weights[i] - {self.weights[i].shape}")
            out = np.dot(last_act, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                last_act = self.relu(out)
            else:
                last_act = out
            self.activations.append(last_act)
        return last_act

    def mse_loss(self, actual: np.ndarray, pred: np.ndarray):
        self.loss = np.mean((actual - pred) ** 2)
        return self.loss

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate=1e-2):
        dA = self.activations[-1] - y
        m = X.shape[0]

        for layer in range(len(self.weights) - 1, -1, -1):
            # print(self.activations[layer].shape,dA.shape)
            dw = np.dot(self.activations[layer].T, dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m
            self.gradient.insert(0, np.linalg.norm(dw))

            if layer > 0:
                # print(dA.shape, self.weights[layer].shape)
                dA = np.dot(dA, self.weights[layer].T)
                dA = dA * self.relu_derivative(self.activations[layer])

            self.weights[layer] -= learning_rate * dw
            self.biases[layer] -= learning_rate * db
        # print(self.gradient)
        return self.gradient
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        1 if x > 0, else 0
        """
        return (x > 0).astype(float)


def train(model: NNModel, X: np.ndarray, y: np.ndarray, epochs=5, lr=0.01):
    model._initilize_weights()

    gradient_norms = []
    losses = []
    for epoch in range(epochs):
        pred = model.forward(X)
        loss = model.mse_loss(y, pred)
        grads = model.backward(X, y, learning_rate=lr)
        gradient_norms.append(np.mean(grads))
        losses.append(loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

    return {
        "losses": losses,
        "gradients": gradient_norms,
        "final_weights": model.weights,
    }


def run_experiment(args):
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, 1)
    y = np.dot(X, true_weights) + 0.1 * np.random.randn(n_samples, 1)

    # Define network architecture
    layers = [20, 64, 32, 16, 1]  
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, 1)
    y = np.dot(X, true_weights) + 0.1 * np.random.randn(n_samples, 1)
    results = {}
    for w_exp in [ "random", "normalize", "xavier", "he"]: # "zero", 
        print(f"Trining with {w_exp} initilization...")
        model = NNModel(init_method=w_exp, layers=layers)
        result = train(model, X, y, epochs=args.epochs, lr=args.lr)
        results[w_exp] = result
    return results

def get_argparser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", default=100)
    args.add_argument("--lr", default=0.01)
    return args.parse_args()

def plot_exp_result(results):
    _, axes = plt.subplots(2,1, figsize=(10,12))
    
    # training loss comparison
    ax1 = axes[0]
    for method, data in results.items():
        ax1.plot(data['losses'], label=method, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # Gradient flow
    ax1 = axes[1]
    for method, data in results.items():
        ax1.plot(data['gradients'], label=method, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('gradients')
    ax1.set_title('gradients Flow')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    args = get_argparser()
    results = run_experiment(args)
    plot_exp_result(results)


if __name__ == "__main__":
    main()
