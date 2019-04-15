import numpy as np


def sigmoid(x):
    return 1/(1 + 2.71828**-x)


def sigmoid_der(sig):
    return sig*(1 - sig)


vec_sig_der = np.vectorize(sigmoid_der())
vec_sig = np.vectorize(sigmoid)


class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        self.activations = []
        self.weights_grad = []
        self.biases_grad = []
        self.activation_grad = []

        for i in range(1, len(layers)):
            self.weights.append(np.random.uniform(-2, 2, (layers[i-1], layers[i])))
            self.biases.append(np.random.uniform(-2, 2, (1, layers[i])))

            self.weights_grad.append(np.zeros(layers[i - 1], layers[i]))
            self.biases_grad.append(np.zeros(1, layers[i]))
            self.activation_grad.append(np.zeros(layers[i - 1]))

    def predict(self, x):
        activations = [x]
        z = []
        for i in range(len(self.weights)):
            z_t = np.dot(self.weights[i], x) - self.biases[i]
            x = vec_sig(z_t)
            activations.append(x)
            z.append(z)
        return x, activations, z

    def eval(self, x, y):
        y_ac, activations, zs = self.predict(x)
        ac_der = 2 * (y_ac - y)

        for i in range(len(self.weights) - 1, 0, -1):
            ac_der = ac_der * vec_sig_der(activations[i + 1])
            self.biases_grad += ac_der
            self.weights_grad += np.dot(ac_der, np.transpose(activations[i]))
            self.activation_grad += np.dot(ac_der, np.transpose(self.weights[i]))

    def apply_grad(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] += lr * self.weights_grad[i]
            self.biases[i] += lr * self.biases_grad[i]
            self.weights_grad[i] = np.zeros(self.weights_grad[i].shape)
            self.biases_grad[i] = np.zeros(self.biases_grad[i].shape)
