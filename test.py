from nn import NeuralNetwork
import numpy as np
import mnist

images = mnist.train_images()
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]) / 255
labels = mnist.train_labels()
labels_oh = np.array([np.zeros(10) for a in labels])
for oh, idx in zip(labels_oh, labels):
    np.put(oh, idx, 1)

nn = NeuralNetwork((images.shape[1], 16, 16, 10))
for i in range(100):
    print(nn.train_batch(images, labels_oh, 0.05, 100))

idx = np.random.choice(images.shape[0], 1)
image = images[idx].reshape(-1)
prediction, _, _ = nn.predict(images[idx].reshape(-1,))
print(np.argmax(prediction))
print(labels[idx])
