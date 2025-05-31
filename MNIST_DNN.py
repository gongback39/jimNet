import numpy as np
from structs.function import loss_functions
from structs.batch import BatchLoader
from structs.optim import Adam, SGD
from structs.layer import fc_layer
from dataset.MNIST import MNIST_data

class model:
    def __init__(self):
        self.fc = fc_layer(28*28, 128, 'sigmoid')
        self.fc2 = fc_layer(128, 64, 'sigmoid')
        self.fc3 = fc_layer(64, 10, 'softmax')
    
    def forward(self, x):
        x = self.fc.forward(x)
        x = self.fc2.forward(x)
        x = self.fc3.forward(x)
        return x
    
    def backward(self, loss, optimizer):
        grad = self.fc3.backward(loss, optimizer)
        grad = self.fc2.backward(grad, optimizer)
        grad = self.fc.backward(grad, optimizer)

def train(model, x, y, criterion, optimizer, epochs=100, batch_size=64):
    loader = BatchLoader(x, y, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in loader:
            y_hat = model.forward(xb)

            loss = criterion[0](yb, y_hat)
            grad_output = criterion[1](yb, y_hat)
            
            model.backward(grad_output, optimizer)
            total_loss += loss * xb.shape[0]

            preds = np.argmax(y_hat, axis=1)
            true = np.argmax(yb, axis=1)
            correct += np.sum(preds == true)
            total += len(true)
            

        avg_loss = total_loss / len(x)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

if __name__ == "__main__":

    X, y = MNIST_data()

    model = model()
    criterion = loss_functions['CE']
    optimizer = Adam(lr=0.001)

    train(model, X, y, criterion, optimizer, epochs=10, batch_size=32)