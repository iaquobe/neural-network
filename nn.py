import numpy as np


def sigmoid(x):
    return 1/(1 + 2.71828**-x)


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


vec_sig_der = np.vectorize(sigmoid_der)
vec_sig = np.vectorize(sigmoid)


class NeuralNetwork:
    def __init__(self, layers=None, file=None):
        self.weights = []
        self.biases = []

        self.weights_grad = []
        self.biases_grad = []
        self.activation_grad = []

        if layers is not None:
            for i in range(1, len(layers)):
                self.weights.append(np.random.uniform(-2, 2, (layers[i], layers[i - 1])))
                self.biases.append(np.random.uniform(-2, 2, (1, layers[i])))

                self.weights_grad.append(np.zeros((layers[i], layers[i - 1])))
                self.biases_grad.append(np.zeros((1, layers[i])))
                self.activation_grad.append(np.zeros((layers[i - 1])))
        elif file is not None:
            self.load(file)
            for weights, biases in zip(self.weights, self.biases):
                self.weights_grad.append(np.zeros(weights.shape))
                self.biases_grad.append(np.zeros(biases.shape))
                self.activation_grad.append(np.zeros(weights.shape[1]))

    def predict(self, x):
        a = [x]
        z = []
        activation = x
        for weights, bias in zip(self.weights, self.biases):
            z_i = np.matmul(weights, activation) + bias
            activation = vec_sig(z_i.flatten())
            a.append(activation)
            z.append(z_i)
        return activation, a, z

    def gradient_descend(self, x, y_l):
        y_p, a, z = self.predict(x)
        z_d = 2 * (y_p - y_l)

        for i in range(len(self.weights) - 1, -1, -1):
            sig_der = vec_sig_der(a[i + 1])
            z_d = z_d * sig_der

            # b_d(i) = z_d(i)
            self.biases_grad[i] += z_d

            # w_d(i) = z_d(i) * a(i +vec_sig_der(a[i + 1]) 1)T
            w_d = np.matmul(z_d[None].T, a[i][None])
            self.weights_grad[i] += w_d

            # a_d(i - i) = z_d(i) *
            t_w = np.transpose(self.weights[i])
            z_d = np.matmul(t_w, z_d)
            self.activation_grad[i] += z_d
        return y_p, np.sum((y_l - y_p)**2)

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

    def train_batch(self, x_s, y_l, lr, size=None):
        if size is not None and size != x_s.shape[0]:
            idx = np.random.choice(x_s.shape[0], size, replace=False)
            x_s = x_s[idx]
            y_l = y_l[idx]
        s_c = 0
        for x, y in zip(x_s, y_l):
            _, c_t = self.gradient_descend(x, y)
            s_c += c_t
        self.apply_grad(lr)
        return s_c / len(x_s)

    def eval(self, x_s, y_l, size=None):
        if size is not None and size != x_s.shape[0]:
            idx = np.random.choice(x_s.shape[0], size, replace=False)
            x_s = x_s[idx]
            y_l = y_l[idx]
        else:
            size = x_s.shape[0]

        correct = 0
        for x, y in zip(x_s, y_l):
            prediction, _, _ = self.predict(x)
            guess = np.argmax(prediction)
            label = np.argmax(y)
            if guess == label:
                correct += 1
            else:
                print("guess={0}, label={1}".format(guess, label))
        return correct / size

    def save(self, file):
        tmp_bias = [x.T for x in self.biases]
        np.savez(file, weights=self.weights, biases=tmp_bias)

    def load(self, file):
        f = np.load(file)
        weights = [m for m in f["weights"]]
        biases = [b.T for b in f["biases"]]
        self.weights = weights
        self.biases = biases
