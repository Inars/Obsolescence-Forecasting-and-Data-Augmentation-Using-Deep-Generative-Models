import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

class TestEncoder(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super(TestEncoder, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % 100 != 0:
            return
        self.current_epoch = self.current_epoch + 1
        encoder_model = Model(inputs=self.model.input,
                            outputs=self.model.get_layer('encoder_output').output)
        encoder_output = encoder_model(self.x_test)
        plt.subplot(4, 3, self.current_epoch)
        plt.scatter(encoder_output[:, 0],
                    encoder_output[:, 1], s=20, alpha=0.8,
                    cmap='Set1', c=self.y_test[0:self.x_test.shape[0]])
        # plt.xlim(-9, 9)
        # plt.ylim(-9, 9)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()

def plot_delta(diff=[], input_dim=0, title=''):
    plt.suptitle(title)
    plt.plot(diff, label='reconstruction delta')
    plt.xticks(np.arange(0, input_dim, 1))
    plt.grid(True)
    plt.show()

def plot_deltas(diff1=[], diff2=[], input_dim=0, title=''):
    plt.suptitle(title)
    plt.plot(diff1, label='before training')
    plt.plot(diff2, label='after training')
    plt.xticks(np.arange(0, input_dim, 1))
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    dataset = 'arrow'
    # dataset = 'phone'
    # dataset = 'sncf'
    path = '../data/' + dataset + '.csv'
    
    # load data
    data = pd.read_csv(path, encoding="latin1")

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # define the autoencoder network model
    # This is the dimension of the original space
    input_dim = X.shape[1]

    # This is the dimension of the latent space (encoding space)
    latent_dim = 2

    encoder = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(latent_dim, activation='relu', name='encoder_output')
    ])

    decoder = Sequential([
        Dense(64, activation='relu', input_shape=(latent_dim,)),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(input_dim, activation=None)
    ])

    autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
    autoencoder.compile(loss='mse', optimizer='adam')

    # plt.figure(figsize=(15,15))
    
    diff_pretrain = np.abs(autoencoder.predict(X) - X)
    diff_pretrain = np.mean(diff_pretrain, axis=0) # average over all samples
    plot_delta(diff_pretrain, input_dim, 'Before training the encoder-decoder')

    # model_history = autoencoder.fit(X, X, epochs=1000, batch_size=250, verbose=0,
    #                                 callbacks=[TestEncoder(X, y)])
    model_history = autoencoder.fit(X, X, epochs=100000, batch_size=250, verbose=0)

    plt.plot(model_history.history["loss"])
    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.show()

    diff_posttrain = np.abs(autoencoder.predict(X) - X)
    diff_posttrain = np.mean(diff_posttrain, axis=0) # average over all samples
    plot_delta(diff_posttrain, input_dim, 'After training the encoder-decoder')
    plot_deltas(diff_pretrain, diff_posttrain, input_dim, 'Reconstruction deltas of the encoder-decoder')

    encoded_x_train = encoder(X)
    plt.figure(figsize=(6,6))
    plt.scatter(encoded_x_train[:, 0], encoded_x_train[:, 1], alpha=.8)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space')
    plt.show()

    # add labels to encoded data
    x_encoded = encoder(X)
    data_encoded = np.column_stack((x_encoded, y))
    data_encoded = np.vstack((['x1', 'x2', 'label'], data_encoded))

    # save encoded data to csv
    np.savetxt('../data/encoded/' + dataset + '.csv', data_encoded, delimiter=",", fmt="%s")
    np.savetxt('../output/ae/' + dataset + '_delta_pretrain.csv', diff_pretrain, delimiter=",", fmt="%s")
    np.savetxt('../output/ae/' + dataset + '_delta_posttrain.csv', diff_posttrain, delimiter=",", fmt="%s")

    # save the model
    autoencoder.save('../models/autoencoders/' + dataset + '.h5')