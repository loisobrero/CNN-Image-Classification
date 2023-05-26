import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    data_dir = './data/raw/cifar-10-batches-py'
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file = 'test_batch'

    # Load training data
    X_train = None
    y_train = []
    for train_file in train_files:
        with open(os.path.join(data_dir, train_file), 'rb') as f:
            train_data = pickle.load(f, encoding='bytes')
        if X_train is None:
            X_train = train_data[b'data']
        else:
            X_train = np.vstack((X_train, train_data[b'data']))
        y_train += train_data[b'labels']
    y_train = np.array(y_train)

    # Load test data
    with open(os.path.join(data_dir, test_file), 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    X_test = test_data[b'data']
    y_test = np.array(test_data[b'labels'])

    # Normalize pixel values
    #X_train = X_train / 255.0
    #X_test = X_test / 255.0

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Mean shift and variance centering
    print(X_train.shape)
    mean = np.mean(X_train, axis = (0,1))
    std = np.std(X_train, axis = (0,1))
    X_train = (X_train-mean)/(std + 1e-7)
    X_test = (X_test-mean)/(std + 1e-7)
    X_val = (X_val-mean)/(std + 1e-7)

    # Save preprocessed data
    processed_dir = './data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, 'train.pkl'), 'wb') as f:
        pickle.dump({'data': X_train, 'labels': y_train}, f)
    with open(os.path.join(processed_dir, 'val.pkl'), 'wb') as f:
        pickle.dump({'data': X_val, 'labels': y_val}, f)
    with open(os.path.join(processed_dir, 'test.pkl'), 'wb') as f:
        pickle.dump({'data': X_test, 'labels': y_test}, f)
    print('Preprocessed data saved in data/processed directory')

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    load_data()



