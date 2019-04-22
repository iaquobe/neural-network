# NeuralNetwork
feed forward neural-network

```
from nn import NeuralNetwork

# initialize from file
nn = NeuralNetwork(file="filename.npz")

# initialize with random values
nn = NeuralNetwork(layers=(784, 64, 32, 10))

# feed forward
network_input = np.random.uniform(-2, 2, 784)
activation = nn.predict(network_input)


# train
network_inputs = np.random.uniform(-2, 2, (20, 784))
network_labels = np.random.uniform(-2, 2, (20, 10))
learning_rate = 0.2
size = 10
nn.train_batch(network_inputs, network_labels, learning_rate, size=size)

# evaluate network: returns correct guesses in percent
nn.eval(network_inputs, network_labels, size=size)

# save network
nn.save("filename.npz")

```
