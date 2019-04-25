from nn import NeuralNetwork
import numpy as np
import mnist
import matplotlib.pyplot as plt


def train(nn):
    for i in range(50):
        cost = nn.train_batch(images, labels_oh, .2, 1000)
        print(cost)


images = mnist.train_images()
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]) / 255
labels = mnist.train_labels()
labels_oh = np.array([np.zeros(10) for a in labels])
for oh, idx in zip(labels_oh, labels):
    np.put(oh, idx, 1)

# nn = NeuralNetwork((images.shape[1], 64, 32, 32, 32, 10))

nn = NeuralNetwork(file="nn3.npz")
sessions = 20
for i in range(sessions):
    train(nn)
    print("session {0}/{1} complete".format(i + 1, sessions))
    nn.save("nn3")


test_images = mnist.test_images()
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]) / 255
test_labels = mnist.test_labels()
test_labels_oh = np.array([np.zeros(10) for a in test_labels])
for oh, idx in zip(test_labels_oh, test_labels):
    np.put(oh, idx, 1)

print("network guessing correct digit {0}% of the time".format(nn.evaluate_network(test_images, test_labels_oh, 1000) * 100))
# '''
