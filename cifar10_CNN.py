import argparse
import numpy as np
from tqdm import tqdm
from dataset.cifar10 import cifar10_train, cifar10_test
from structs.function import loss_functions
from structs.batch import BatchLoader
from structs.optim import Adam
from structs.layer import fc_layer, conv2d_layer, max_pool2d_layer, Flatten

class Model:
    def __init__(self):
        self.conv1 = conv2d_layer(3, 8, 3, 1, 1, 'ReLU') # input: 32x32x3, output: 32x32x8, kernel size: 3x3, stride: 1, padding: 1
        self.pool1 = max_pool2d_layer(2, 2) # input: 32x32x8, output: 16x16x8, kernel size: 2x2, stride: 2
        self.conv2 = conv2d_layer(8, 16, 3, 1, 1, 'ReLU') # input: 16x16x8, output: 16x16x16, kernel size: 3x3, stride: 1, padding: 1
        self.pool2 = max_pool2d_layer(2, 2) # input: 16x16x16, output: 8x8x16, kernel size: 2x2, stride: 2
        self.flatten = Flatten() # 2D -> 1D flatten
        self.fc1 = fc_layer(8 * 8 * 16, 64, 'ReLU') # input: 8*8*16, output: 64
        self.fc2 = fc_layer(64, 32, 'ReLU') # input: 64, output: 32
        self.fc3 = fc_layer(32, 10, 'softmax') # input: 32, output: 10

    def forward(self, x):
        # feed forward
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, loss, optimizer):
        # backpropagation
        grad = self.fc3.backward(loss, optimizer)
        grad = self.fc2.backward(grad, optimizer)
        grad = self.fc1.backward(grad, optimizer)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.conv2.backward(grad, optimizer)
        grad = self.pool1.backward(grad)
        grad = self.conv1.backward(grad, optimizer)


def train(model, x, y, criterion, optimizer, epochs=100, batch_size=64):
    loader = BatchLoader(x, y, batch_size=batch_size, shuffle=True) # batch loader

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        # tqdm으로 배치 진행률 표시
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for xb, yb in pbar:
            # feed forward
            y_hat = model.forward(xb)

            # loss 계산
            loss = criterion[0](yb, y_hat)
            grad_output = criterion[1](yb, y_hat)
            
            # backpropagation
            model.backward(grad_output, optimizer)

            # loss 계산
            total_loss += loss * xb.shape[0]
            
            # accuracy 계산
            preds = np.argmax(y_hat, axis=1)
            true = np.argmax(yb, axis=1)
            correct += np.sum(preds == true)
            total += len(true)
        
        # 평균 loss 계산, accuracy 계산
        avg_loss = total_loss / len(x)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

def eval(model, x, y, criterion, batch_size=64):
    loader = BatchLoader(x, y, batch_size=batch_size, shuffle=False)

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Evaluation", leave=False)

    for xb, yb in pbar:
        # feed forward
        y_hat = model.forward(xb)

        # loss 계산
        loss = criterion[0](yb, y_hat)
        total_loss += loss * xb.shape[0]

        # accuracy 계산
        preds = np.argmax(y_hat, axis=1)
        true = np.argmax(yb, axis=1)
        correct += np.sum(preds == true)
        total += len(true)

    # 평균 loss 계산, accuracy 계산
    avg_loss = total_loss / len(x)
    accuracy = correct / total
    print(f"[Evaluation] Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

    return accuracy, avg_loss

if __name__ == "__main__":
    # 실행 예시: python model_train_test.py --epochs 10 --batch_size 32 --lr 0.001
    parser = argparse.ArgumentParser(description="Train CIFAR CNN Model")
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    train_data, train_label = cifar10_train()
    test_data, test_label = cifar10_test()

    model = Model()
    criterion = loss_functions['CE']
    optimizer = Adam(lr=args.lr)

    train(model, train_data, train_label, criterion, optimizer, epochs=args.epochs, batch_size=args.batch_size)
    eval(model, test_data, test_label, criterion)
