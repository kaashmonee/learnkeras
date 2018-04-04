# tutorial from elitedatascience.com -- including comments for my benefit

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

from keras.utils import np_utils

# importing mnist database
from keras.datasets import mnist

def main():

    # print("minst.load_data(): ", mnist.load_data())
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    # This returns (60000, 28, 28), which is the shape of the X_train from mnist
    # This means that there are 60000 samples in data set and images are 20 x 28
    # pixels

    # showing the plot (ie, the first 28 x 28 image)
    print("reaching here!")
    plt.show(X_train[0])
    print("not reaching here")
    plt.show()
    print("yo what the fuck??")
    plt.savefig("mypic.png")

if __name__ == "__main__":
    main()