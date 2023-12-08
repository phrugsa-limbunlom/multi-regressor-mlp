import numpy as np
import pandas as pd

from my_trained_model.dl_training_model_no_batch import Sequential, SGD, RootMeanSquaredError, Layer


class NeuralNetHolder:

    def __init__(self):
        super().__init__()

        #initialize model
        self.model = Sequential()
        self.model.add(Layer(2))  # Input layer
        self.model.add(Layer(4, activation="sigmoid"))  # Hidden layer
        self.model.add(Layer(1))  # Output layer
        self.model.compiler(optimizer=SGD(learning_rate=1e-6, momentum=8e-5), loss=RootMeanSquaredError())
        self.weight_input_hidden, self.weight_hidden_output, self.bias_hidden, self.bias_output = self.model.load(
            "my_trained_model/model_saved.npz")

        # initialize min-max values
        df = pd.read_csv("my_trained_model/ce889_dataCollection_15k.csv", names=['col1', 'col2', 'col3', 'col4'])

        x = df[['col1', 'col2']].to_numpy()
        y = df[['col3', 'col4']].to_numpy()

        self.max_x1 = np.max(x[:, 0])
        self.min_x1 = np.min(x[:, 0])

        self.max_x2 = np.max(x[:, 1])
        self.min_x2 = np.min(x[:, 1])

        self.max_y1 = np.max(y[:, 0])
        self.min_y1 = np.min(y[:, 0])

        self.max_y2 = np.max(y[:, 1])
        self.min_y2 = np.min(y[:, 1])

    def normalize(self, input_row):
        input = input_row.split(",")
        x1 = float(input[0])
        x2 = float(input[1])

        # min-max normalization [0,1]
        x1_norm = (x1 - float(self.min_x1)) / (float(self.max_x1) - float(self.min_x1))
        x2_norm = (x2 - float(self.min_x2)) / (float(self.max_x2) - float(self.min_x2))
        print("x1 ", x1)
        print("x2 ", x2)
        print("x1_norm ", x1_norm)
        print("x2_norm ", x2_norm)
        return x1_norm, x2_norm

    def denormalize(self, pred):
        y1_norm = pred[0, 0]
        y2_norm = pred[0, 1]
        y1 = (y1_norm * (self.max_y1 - self.min_y1)) + self.min_y1
        y2 = (y2_norm * (self.max_y2 - self.min_y2)) + self.min_y2

        print("y1_norm ", y1_norm)
        print("y2_norm ", y2_norm)
        print("y1 ", y1)
        print("y2 ", y2)
        return y1, y2

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        x1, x2 = self.normalize(input_row)  # x1 = x distance to target, x2 = y distance to target

        pred = self.model.predict(np.array([x1, x2]))
        y1, y2 = self.denormalize(pred)  # y1 = velocity x, y2 = velocity y
        return y1, y2
