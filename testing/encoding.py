import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def main():
    # load data
    # DATASET_NAMES = ['moons','arrow','phone']
    DATASET_NAMES = ['arrow','phone']
    
    fig = go.Figure()

    for DATASET_NAME in DATASET_NAMES:
        data = pd.read_csv('../data/' + DATASET_NAME + '.csv', sep=",", encoding='latin1')

        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        input_dim = X.shape[1]

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        # X = scaler.inverse_transform(data)

        # Define the model
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

        rmse_preencoding = np.sqrt(np.mean(np.power(X - autoencoder.predict(X), 2), axis=0))

        # Load saved model
        autoencoder = tf.keras.models.load_model('../models/autoencoders/' + DATASET_NAME + '.h5')
        X_encoded = autoencoder.predict(X)
        # rmse between each column of X and X_encoded
        rmse_postencoding = np.sqrt(np.mean(np.power(X - X_encoded, 2), axis=0))

        print ("=====================================")
        print(DATASET_NAME + " RMSE before encoding:", rmse_preencoding)
        print(DATASET_NAME + " RMSE after encoding:", rmse_postencoding)
        print(DATASET_NAME + " RMSE average before encoding:", np.mean(rmse_preencoding))
        print(DATASET_NAME + " RMSE average after encoding:", np.mean(rmse_postencoding))

        x_axis = np.arange(0, input_dim, 1)

        # Plot RMSE before and after encoding
        if DATASET_NAME == "arrow":
            fig.add_trace(go.Scatter(x=x_axis, y=rmse_preencoding,
                                mode='lines',
                                name=DATASET_NAME + ' before',
                                line=dict(color=px.colors.qualitative.Plotly[0], width=4)))
            fig.add_trace(go.Scatter(x=x_axis, y=rmse_postencoding,
                                mode='lines',
                                name=DATASET_NAME + ' after',
                                line=dict(color=px.colors.qualitative.Plotly[1], width=4)))
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=rmse_preencoding,
                                mode='lines',
                                name=DATASET_NAME + ' before',
                                line=dict(color=px.colors.qualitative.Plotly[2], width=4, dash='dot')))
            fig.add_trace(go.Scatter(x=x_axis, y=rmse_postencoding,
                                mode='lines',
                                name=DATASET_NAME + ' after',
                                line=dict(color=px.colors.qualitative.Plotly[3], width=4, dash='dot')))
    
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="RMSE",
        font=dict(family="Times New Roman")
    )
    fig.show()
    fig.write_image("../media/encoding.png")
        
    # Make a prediction
    # prediction = model.predict(input_data)
    
    # Print the prediction
    # print("Prediction:", prediction)


if __name__ == "__main__":
    main()