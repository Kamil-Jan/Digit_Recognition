import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def main():
    # Load train data
    digits = pd.read_csv("data/train.csv")
    data = digits.loc[:, digits.columns != "label"] # get input data
    target = digits["label"] # get output data

    # Convert DataFrame.Series to numpy array
    data = data.to_numpy()
    target = target.to_numpy().ravel() # convert row vector into col vector

    # Divide data by 255 for a smaller range of features
    data = data / 255.0
    data = np.expand_dims(data, 1)

    # Create a model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1, 784)),
        keras.layers.Dense(125, activation="relu"),
        keras.layers.Dense(10)
    ])

    # Compile a model
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # Train
    model.fit(data, target, epochs=30)

    # Save a model
    filename = "model.h5"
    model.save(filename)
    print("Done!")

if __name__ == "__main__":
    main()

