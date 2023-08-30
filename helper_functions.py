import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch




def predict(X, network):
    output = X.T
    for layer in network:
        output = layer.forward(output)
    return output.argmax(axis=0)



def decision_surface(network, predict, ax, X, y, n_classes):
    """
    A function to draw the decision surface for a given model in a 2-D plot.

    Inputs:
        - classifier: a classifier object
        - ax: An matplotlib.pyplot axes object
        - X: the whole dataset used in model (including both train and test sets).
        - n_classes: the number of classes

    Outputs:
        - An plot containing the decision surface for the given classifier.
    """
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    yhat = predict(grid, network)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    ax.contourf(xx, yy, zz, cmap='Set3', alpha=0.7)
    # plot the datapoints
    for i in range(n_classes):
        class_i = X[y == i]
        ax.scatter(class_i[:, 0], class_i[:, 1], label=f'class {i}')
    ax.legend()


def decision_surface_torch(model, ax, X, y, n_classes):
    """
    A function to draw the decision surface for a given model in a 2-D plot.

    Inputs:
        - classifier: a classifier object
        - ax: An matplotlib.pyplot axes object
        - X: the whole dataset used in model (including both train and test sets).
        - n_classes: the number of classes

    Outputs:
        - An plot containing the decision surface for the given classifier.
    """
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))



    grid = torch.tensor(grid).type(torch.float)

    model.eval()
    with torch.inference_mode():
        train_scores = model(grid)
    
    yhat = train_scores.numpy().argmax(axis=1) # go from scores -> prediction labels
    
    # make predictions for the grid
    # yhat = predict(grid, network)
    
    
    
    
    
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    ax.contourf(xx, yy, zz, cmap='Set3', alpha=0.7)
    # plot the datapoints
    for i in range(n_classes):
        class_i = X[y == i]
        ax.scatter(class_i[:, 0], class_i[:, 1], label=f'class {i}')
    ax.legend()


def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        return X, Y

def load_cifar10(path):
    """Load all batches from CIFAR-10 dataset."""
    # load train data batch
    xs = []
    ys = []
    for i in range(1, 6):
        filename = os.path.join(path, f'data_batch_{i}')
        X, y = load_cifar10_batch(filename)
        xs.append(X)
        ys.append(y)
    Xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    del X, y

    # load test data batch
    Xte, yte = load_cifar10_batch(os.path.join(path, 'test_batch'))
    return Xtr, ytr, Xte, yte

