import pickle
import pandas as pd
import numpy as np

# pickle 파일을 읽어와서 dictionary로 변환
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# batch를 DataFrame으로 변환
def makeDf(batch):
    df = pd.DataFrame({'data': list(batch[b'data']),  'labels': list(batch[b'labels'])})
    return df

# 1차원 리스트의 이미지 데이터를 3차원으로 변환
def matchShape(batch):
    matched_data = []
    label = []
    for j in range(len(batch['data'])):
        data = batch['data'].values[j]

        R = data[0:1024].reshape(32, 32)
        G = data[1024:2048].reshape(32, 32)
        B = data[2048:].reshape(32, 32)

        matched_data.append(np.dstack((R, G, B)))
        zeros = [0]*10
        zeros[batch['labels'][j]] = 1
        label.append(zeros)
    return matched_data, label

# call train data
def cifar10_train():
    batch = []
    batch_num = 5
    for i in range(batch_num):
        batch.append(unpickle("./cifar-10-batches-py/data_batch_" + str((i+1))))

    for i in range(batch_num):
        batch[i] = makeDf(batch[i])

    train_data = []
    train_label = []
    for i in range(batch_num):
        data, label = matchShape(batch[i])
        train_data += data
        train_label += label

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data = train_data / 255. # 정규화

    return train_data, train_label

# call test data
def cifar10_test():
    test_batch = unpickle("./cifar-10-batches-py/test_batch")

    test_batch = makeDf(test_batch)
    
    test_data = []
    test_label = []
    test_data, test_label = matchShape(test_batch)
    test_data = np.array(test_data) / 255. # 정규화
    
    return test_data, test_label