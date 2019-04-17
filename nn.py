import numpy as np


def sigmoid(x):
    return 1/(1 + 2.71828**-x)


def sigmoid_der(sig):
    return sig*(1 - sig)


vec_sig_der = np.vectorize(sigmoid_der)
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
            self.weights.append(np.random.uniform(-2, 2, (layers[i], layers[i - 1])))
            self.biases.append(np.random.uniform(-2, 2, (1, layers[i])))

            self.weights_grad.append(np.zeros((layers[i], layers[i - 1])))
            self.biases_grad.append(np.zeros((1, layers[i])))
            self.activation_grad.append(np.zeros((layers[i - 1])))

    def predict(self, x):
        a = [x]
        z = []
        a_i = x
        for i in range(len(self.weights)):
            z_i = np.matmul(self.weights[i], a_i) + self.biases[i]
            a_i = vec_sig(z_i.flatten())
            a.append(a_i)
            z.append(z_i)
        return a_i, a, z

    def eval(self, x, y_l):
        y_p, a, z = self.predict(x)
        z_d = 2 * (y_p - y_l)

        for i in range(len(self.weights) - 1, -1, -1):
            z_d = z_d * vec_sig_der(a[i + 1])

            # b_d(i) = z_d(i)
            self.biases_grad[i] += z_d

            # w_d(i) = z_d(i) * a(i + 1)T
            w_d = np.matmul(z_d[None].T, a[i][None])
            self.weights_grad[i] += w_d

            # a_d(i - i) = z_d(i) *
            t_w = np.transpose(self.weights[i])
            z_d = np.matmul(t_w, z_d)
            self.activation_grad[i] += z_d
        return y_p, np.sum(y_l - y_p)

    def apply_grad(self, lr):
        s = 0
        for i in range(len(self.weights)):
            s += np.sum(self.weights[i])
            s += np.sum(self.biases[i])
        lr = lr / abs(s)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.weights_grad[i]
            self.biases[i] -= lr * self.biases_grad[i]
            self.weights_grad[i] = np.zeros(self.weights_grad[i].shape)
            self.biases_grad[i] = np.zeros(self.biases_grad[i].shape)

    def train_batch(self, x_s, y_l, lr, size):
        if size != x_s.shape[0]:
            idx = np.random.choice(len(x_s), size, replace=False)
            x_s = x_s[idx]
            y_l = y_l[idx]
        s_c = 0
        for x, y in zip(x_s, y_l):
            _, c_t = self.eval(x, y)
            s_c += c_t
        self.apply_grad(lr)
        return s_c / len(x_s)
