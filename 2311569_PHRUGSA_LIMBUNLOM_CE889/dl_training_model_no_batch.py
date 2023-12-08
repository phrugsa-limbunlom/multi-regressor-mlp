import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import warnings
import sys

# suppress warnings
warnings.filterwarnings('ignore')


class SGD:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum


class RootMeanSquaredError:

    @staticmethod
    def mse(y_act, y_pred):
        return np.mean((y_act - y_pred) ** 2)

    def rmse(self, y_act, y_pred):
        return np.sqrt(self.mse(y_act, y_pred))


class Layer:
    def __init__(self, neuron, activation=None):
        self.neuron = neuron
        self.activation = activation


class Sequential:
    def __init__(self):
        self.loss = None
        self.optimizer = None
        self.layers = []
        self.training_losses = []
        self.validation_losses = []
        self.count_feedforward = 1

    def add(self, layer):
        self.layers.append(layer)

    def compiler(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.optimizer.learning_rate * x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    def feedforward(self, x, w, b):
        if self.count_feedforward >= len(self.layers):
            self.count_feedforward = 1
        if self.layers[self.count_feedforward].activation == "sigmoid":
            self.count_feedforward += 1
            return self.sigmoid(x @ w + b)
        else:
            self.count_feedforward += 1
            return x @ w + b

    def gradient_hidden(self, gradient_output, hidden_layer_neurons):
        sum_of_previous_gradients_and_weights = gradient_output @ self.weight_hidden_output.T
        gradient_hidden = self.optimizer.learning_rate * self.dsigmoid(
            hidden_layer_neurons) * sum_of_previous_gradients_and_weights
        return gradient_hidden

    @staticmethod
    def gradient_output(y_act, y_pred):
        return y_act - y_pred

    def calculate_dweight(self, gradient, neuron, prev_weight):
        return self.optimizer.learning_rate * (neuron.T @ gradient) + self.optimizer.momentum * prev_weight

    def calculate_dbias(self, gradient, prev_bias):
        return np.sum(gradient, axis=0,
                      keepdims=True) * self.optimizer.learning_rate + self.optimizer.momentum * prev_bias

    def update_weights_and_bias(self, dw1, dw2, b1, b2):
        self.weight_hidden_output += dw2
        self.weight_input_hidden += dw1
        self.bias_output += b2
        self.bias_hidden += b1

    def update_prev_weights_and_bias(self, dw1, dw2):
        self.prev_weight_hidden_output = dw2
        self.prev_weight_input_hidden = dw1
        self.prev_bias_output = self.bias_output
        self.prev_bias_hidden = self.bias_hidden

    def initialize_weights_and_biases(self, input_size, hidden_size, output_size):
        # initialize weights and biases
        np.random.seed(42)
        self.weight_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weight_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # initialize previous weights and biases
        self.prev_weight_input_hidden = np.zeros_like(self.weight_input_hidden)
        self.prev_bias_hidden = np.zeros_like(self.bias_hidden)
        self.prev_weight_hidden_output = np.zeros_like(self.weight_hidden_output)
        self.prev_bias_output = np.zeros_like(self.bias_output)

    def fit(self, x_train, y_train, x_val, y_val, epochs):

        input_neurons_size = self.layers[0].neuron
        hidden_neurons_size = self.layers[1].neuron
        output_neurons_size = self.layers[2].neuron

        self.initialize_weights_and_biases(input_neurons_size, hidden_neurons_size, output_neurons_size)

        # early stopping parameters
        max_failed_threshold = 10
        current_failed_threshold = 0

        minimum_loss = 1.0  # Initialize with a max value

        # process epoch
        for epoch in range(epochs):

            start_time = time.time()

            shuffle_idx = np.random.permutation(len(x_train))
            X_train_shuffled, y_train_shuffled = x_train[shuffle_idx, :], y_train[shuffle_idx, :]

            # feedforward
            hidden_layer_neurons = self.feedforward(X_train_shuffled, self.weight_input_hidden,
                                                    self.bias_hidden)

            y_pred = self.feedforward(hidden_layer_neurons, self.weight_hidden_output, self.bias_output)

            # backward propagation on output to hidden layer
            gradient_output = self.gradient_output(y_train, y_pred)  # linear

            d_weight_hidden_output = self.calculate_dweight(gradient_output, hidden_layer_neurons,
                                                            self.prev_weight_hidden_output)

            d_bias_output = self.calculate_dbias(gradient_output, self.prev_bias_output)

            # backward propagation on hidden to input layer
            gradient_hidden = self.gradient_hidden(gradient_output, hidden_layer_neurons)

            d_weight_input_hidden = self.calculate_dweight(gradient_hidden, X_train_shuffled,
                                                           self.prev_weight_input_hidden)

            d_bias_hidden = self.calculate_dbias(gradient_hidden, self.bias_hidden)

            # update weights and biases
            self.update_weights_and_bias(d_weight_input_hidden, d_weight_hidden_output, d_bias_hidden,
                                         d_bias_output)

            # save delta weights and biases for next epoch
            self.update_prev_weights_and_bias(d_weight_input_hidden, d_weight_hidden_output)

            # loss of training set per one epoch
            training_loss = self.loss.rmse(y_train_shuffled, y_pred)
            self.training_losses.append(training_loss)

            # loss of validation set per one epoch
            y_val_pred = self.predict(x_val)
            validation_loss = self.loss.rmse(y_val, y_val_pred)
            self.validation_losses.append(validation_loss)

            # early stopping criteria
            if validation_loss < minimum_loss:
                minimum_loss = validation_loss
                current_failed_threshold = 0
            else:
                current_failed_threshold += 1

            # stop fitting the model for current hyperparameters
            # if a number of the current loss greater than the previous loss
            # exceeding the threshold limit (current maximum is 5)
            if current_failed_threshold >= max_failed_threshold:
                logging.info(f"Early stopping: No improvement")
                return False

            end_time = time.time()

            time_per_epoch = (end_time - start_time)

            logging.info(f"Epoch {epoch + 1}/{epochs} - Time: {time_per_epoch:.2f} seconds")
            logging.info(
                f"[===========================================================] Training Loss: {training_loss:.8f}, Validation Loss: {validation_loss:.8f}")

        return True

    def predict(self, x):

        hidden_layer_neurons = self.sigmoid(x @ self.weight_input_hidden + self.bias_hidden)

        predicted = hidden_layer_neurons @ self.weight_hidden_output + self.bias_output

        return predicted

    def evaluate(self, x_test, y_test):

        y_pred = model.predict(x_test)

        return self.loss.rmse(y_test, y_pred)

    def save(self, path):
        np.savez(path, weights1=self.weight_input_hidden, weights2=self.weight_hidden_output, bias1=self.bias_hidden,
                 bias2=self.bias_output)

    def load(self, path):
        data = np.load(path)
        self.weight_input_hidden = data['weights1']
        self.weight_hidden_output = data['weights2']
        self.bias_hidden = data['bias1']
        self.bias_output = data['bias2']
        return self.weight_input_hidden, self.weight_hidden_output, self.bias_hidden, self.bias_output


class Model_Selection:
    @staticmethod
    def train_test_split(x, y, test_size, shuffle=True):
        if shuffle:
            train_size = 1 - test_size
            # Shuffle the training set
            train_index = np.random.permutation(int(round(x.shape[0] * train_size)))
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            # Filter out validation set from training set
            val_index = [i for i in range(x.shape[0]) if i not in train_index]
            x_val = x[val_index, :]
            y_val = y[val_index, :]
            return x_train, y_train, x_val, y_val

        # split data to train and test sets
        train_size = 1 - test_size
        split_training_index = int(round(len(x) * train_size))
        x_train, y_train = x[:split_training_index], y[:split_training_index]
        x_test, y_test = x[split_training_index:], y[split_training_index:]

        return x_train, y_train, x_test, y_test


def preprocessing(dataframe):
    df = dataframe.copy()

    # drop rows if having null values
    if df.isna().values.any():
        df.dropna()

    # min-max normalization [0,1]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    x = df[['col1', 'col2']].to_numpy()
    y = df[['col3', 'col4']].to_numpy()

    return x, y


def train_model(path, x_train, y_train, x_val, y_val):
    # hyperparameters grid for tuning
    learning_rates = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    momentum = [0.9, 0.95, 0.8, 8e-2, 8e-3, 8e-5]

    final_lr = 0.0
    final_momentum = 0.0

    final_model = None

    for lr in learning_rates:
        terminate = False
        for m in momentum:

            logging.info(f"Start tuning hyperparameters : learning_rate = {lr} and momentum = {m}")

            model = Sequential()
            model.add(Layer(x_train.shape[1]))  # Input layer
            model.add(Layer(hidden_neuron, activation="sigmoid"))  # Hidden layer
            model.add(Layer(y_train.shape[1]))  # Output layer

            sgd = SGD(learning_rate=lr, momentum=m)
            rmse = RootMeanSquaredError()

            model.compiler(optimizer=sgd, loss=rmse)

            fit = model.fit(x_train, y_train, x_val, y_val, epochs=100)

            if fit:
                logging.info("No early stopping!")
                final_lr = lr
                final_momentum = m
                final_model = model
                model.save(path)
                terminate = True

        if terminate:
            break

    logging.info(f"Final learning rate is {final_lr}")
    logging.info(f"Final momentum is {final_momentum}")

    return final_model


def test_model(model, x_test, y_test):
    loss = model.evaluate(x_test, y_test)

    logging.info(f"RMSE on test set is {loss:.8f}")


def load_model(path):
    w1, w2, b1, b2 = model.load(path + ".npz")

    logging.info("Saved weight input hidden is ")
    logging.info(w1)
    logging.info("Saved weight hidden output is ")
    logging.info(w2)
    logging.info("Saved bias hidden is ")
    logging.info(b1)
    logging.info("Saved bias output is ")
    logging.info(b2)


def visualization(model):
    optimal_training_loss = min(model.training_losses)
    optimal_training_epoch = model.training_losses.index(min(model.training_losses))

    optimal_validation_loss = min(model.validation_losses)
    optimal_validation_epoch = model.validation_losses.index(min(model.validation_losses))

    logging.info(f"Optimal point for Training set is Loss :{optimal_training_loss:.8f} at {optimal_training_epoch + 1}")
    logging.info(
        f"Optimal point for Validation set is Loss :{optimal_validation_loss:.8f} at {optimal_validation_epoch + 1}")

    plt.scatter(optimal_training_epoch, optimal_training_loss, color='r', label=f'Optimal Point for Training set')
    plt.scatter(optimal_validation_epoch, optimal_validation_loss, color='b', label=f'Optimal Point for Validation set')

    plt.plot(model.training_losses, label='Training Loss')
    plt.plot(model.validation_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.legend()
    plt.savefig('Loss of 15K')
    plt.show()


if __name__ == "__main__":
    # parameters region
    file_name = sys.argv[1]
    log_name = sys.argv[2]
    input_neuron = int(sys.argv[3])
    hidden_neuron = int(sys.argv[4])
    output_neuron = int(sys.argv[5])

    # file_name = "15k/ce889_dataCollection_15k.csv"
    # log_name = "activity_hidden4.log"
    # input_neuron = 2
    # hidden_neuron = 4  # 2/3(in+out)
    # output_neuron = 2

    save_path = 'model_saved'
    # end parameters region

    logging.basicConfig(filename=log_name, level=logging.INFO)

    logging.info(f"File name is {file_name}")

    logging.info(
        f"Input neuron size : {input_neuron}, Hidden neuron size : {hidden_neuron}, Output neuron size : {output_neuron}")

    # preprocessing data
    df = pd.read_csv(file_name, names=['col1', 'col2', 'col3', 'col4'])

    x, y = preprocessing(df)
    # end preprocessing data

    logging.info(f"Features size : {len(x)}, Labels size : {len(y)}")

    validation_size = 0.15
    test_size = 0.15

    x_train, y_train, x_test, y_test = Model_Selection().train_test_split(x, y, test_size, False)

    x_train, y_train, x_val, y_val = Model_Selection().train_test_split(x_train, y_train,
                                                                        validation_size / (1 - test_size))

    logging.info(
        f"X train size : {len(x_train)}, y train size : {len(y_train)}, X test size : {len(x_test)}, y test size : {len(y_test)}")

    logging.info(
        f"X train size : {len(x_train)}, y train size : {len(y_train)}, X validation size : {len(x_val)}, y validation size : {len(y_val)}")

    model = train_model(save_path, x_train, y_train, x_val, y_val)

    test_model(model, x_test, y_test)

    load_model(save_path)

    visualization(model)
