# tutorial from elitedatascience.com -- including comments for my benefit
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# "augmented" with my own documentation
import numpy as np
from imread import imread, imsave
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
    # for some stupid fucking reason matplotlib doesn't work
    # matplotlib not working atm
    # print("reaching here!")
    # plt.show(X_train[0])
    # print("not reaching here")
    # plt.show()
    # print("yo what the fuck??")
    # plt.savefig("mypic.png")

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # printing X's dimensions once again
    # Split data into testing and training data
    print(X_train.shape)

    # converting datatype to float32 and normalizing data values to range [0, 1]
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # normalizes 32 bit float
    X_train /= 255
    X_test /= 255

    print("ytrainshape:", y_train.shape)

    print(y_train[:10])

    # what we saw above isn't what we want. we want something that 
    # not entirely sure what these lines here do
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    print(Y_train.shape)


    # building neural network
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(28, 28, 1)))
    # adding more layers to model like we're building legos xD
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.25))

    # adding a fully connected layer and output layer
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)

    score = model.evaluate(X_test, Y_test, verbose = 0)

    test_image(model)

    # print(model.output_shape)

def test_image(model):
    yes_no = True
    while yes_no:
        path = input("Please enter the path: ")
        im = imread(path) 
        im = im/255
        pr = model.predict_classes(im.reshape((28, 28, 1, 1)))
        print("Result: ", pr)
        
        yes_no = True if (input("Would you like to continue? [Y/n] ") == "y" or "Y") else False



if __name__ == "__main__":
    main()