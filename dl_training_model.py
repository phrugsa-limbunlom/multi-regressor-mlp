import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys

#suppress warnings
warnings.filterwarnings('ignore')

class SGD():
    def __init__(self,learning_rate,momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum

class RootMeanSquaredError():
    def mse(self,y_act,y_pred):
        return np.mean((y_act - y_pred) ** 2)
    def rmse(self,y_act,y_pred):
        return np.sqrt(self.mse(y_act, y_pred))
    

class Layer():
    def __init__(self,neuron,activation = None):
        self.neuron = neuron
        self.activation = activation

class Sequential():
    def __init__(self):
        self.layers = []
        self.training_losses = []
        self.validation_losses = []
        self.count_feedforward = 1

    def add(self, layer):
        self.layers.append(layer)

    def compiler(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-self.optimizer.learning_rate * x))

    def dsigmoid(self,x):
        return x * (1 - x)

    def feedforward(self,x,w,b):
        if self.count_feedforward >= len(self.layers):
            self.count_feedforward = 1
        if self.layers[self.count_feedforward].activation == "sigmoid" :
            self.count_feedforward += 1
            return self.sigmoid(x @ w + b)
        else:
            self.count_feedforward += 1
            return x @ w + b

    def initialize_weights_and_biases(self, input_size, hidden_size, output_size):
        #initialize weights and biases
        np.random.seed(42)
        self.weight_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weight_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        #initialize previous weights and biases
        self.prev_weight_input_hidden = np.zeros_like(self.weight_input_hidden)
        self.prev_bias_hidden = np.zeros_like(self.bias_hidden)
        self.prev_weight_hidden_output = np.zeros_like(self.weight_hidden_output)
        self.prev_bias_output = np.zeros_like(self.bias_output)


    def fit(self,x,y,epochs):

        input_layer_neurons_size = self.layers[0].neuron
        hidden_layer_neurons_size = self.layers[1].neuron
        output_layer_neurons_size = self.layers[2].neuron

        self.initialize_weights_and_biases(input_layer_neurons_size,hidden_layer_neurons_size,output_layer_neurons_size)

        prev_val_loss = 0.0 #initialize total loss

        #early stopping parameters
        max_epochs = 5
        current_epochs = 0

        minimum_loss = 1  # Initialize with a max value

        for epoch in range(epochs):

            start_time = time.time()

            x_train,y_train,x_val,y_val = Model_Selection().train_test_split(x,y,validation_size/(1-test_size))

            #feedforward
            hidden_layer_neurons = self.feedforward(x_train, self.weight_input_hidden, self.bias_hidden)

            y_pred = self.feedforward(hidden_layer_neurons, self.weight_hidden_output, self.bias_output)

            #calculate loss of training set
            training_loss = self.loss.rmse(y_train, y_pred)
            self.training_losses.append(training_loss)

            #backward propagation
            error = y_train - y_pred
            gradient_output = error #linear

            d_weight_hidden_output = self.optimizer.learning_rate * (hidden_layer_neurons.T @ gradient_output) + self.optimizer.momentum * self.prev_weight_hidden_output

            sum_of_previous_gradients_and_weights = gradient_output @ self.weight_hidden_output.T
            gradient_hidden = self.optimizer.learning_rate * self.dsigmoid(hidden_layer_neurons) * sum_of_previous_gradients_and_weights

            d_weight_input_hidden = self.optimizer.learning_rate * (x_train.T @ gradient_hidden) + self.optimizer.momentum * self.prev_weight_input_hidden

            #update weight
            self.weight_hidden_output += d_weight_hidden_output

            self.weight_input_hidden += d_weight_input_hidden

            #update bias
            d_bias_output = np.sum(gradient_output, axis=0, keepdims=True) * self.optimizer.learning_rate + self.optimizer.momentum * self.prev_bias_output
            self.bias_output += d_bias_output

            d_bias_hidden = np.sum(gradient_hidden, axis=0, keepdims=True) * self.optimizer.learning_rate + self.optimizer.momentum * self.prev_bias_hidden
            self.bias_hidden += d_bias_hidden


            #save delta weights and biases for next epoch
            self.prev_weight_hidden_output = d_weight_hidden_output
            self.prev_weight_input_hidden = d_weight_input_hidden
            self.prev_bias_output = self.bias_output
            self.prev_bias_hidden = self.bias_hidden


            #loss of validation set
            y_val_pred = self.predict(x_val)
            validation_loss = self.loss.rmse(y_val, y_val_pred)
            self.validation_losses.append(validation_loss)
            prev_val_loss = validation_loss

            if prev_val_loss < minimum_loss:
               minimum_loss = prev_val_loss
               current_epochs = 0
            else:
               current_epochs += 1

            if current_epochs >= max_epochs:  #stop fitting the model for current hyperparameters
              print(f"Early stopping: No improvement")
              return False

            end_time = time.time()

            time_per_epoch = (end_time - start_time)
             
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Time: {time_per_epoch:.2f} seconds")
                print(f"[===========================================================] Training Loss: {training_loss:.8f}, Validation Loss: {validation_loss:.8f}")

        return True


    def predict(self,x):

        hidden_layer_neurons = self.feedforward(x, self.weight_input_hidden, self.bias_hidden)

        predicted =  self.feedforward(hidden_layer_neurons, self.weight_hidden_output, self.bias_output)

        return predicted

    def save(self, path):
        np.savez(path, weights1=self.weight_input_hidden, weights2=self.weight_hidden_output, bias1=self.bias_hidden, bias2=self.bias_output)

    def load(self, path):
        data = np.load(path)
        self.weight_input_hidden = data['weights1']
        self.weight_input_hidden = data['weights1']
        self.bias_hidden = data['bias1']
        self.bias_output = data['bias2']


class Model_Selection():
  def train_test_split(self,x,y,test_size,shuffle=True):
      if shuffle:
        train_size = 1-test_size
        # Shuffle the training set
        train_index = np.random.permutation(int(round(x.shape[0]*train_size)))
        x_train = x[train_index,:]
        y_train = y[train_index,:]
        # Filter out validation set from training set
        val_index = [i for i, x in enumerate(x) if x_train.tolist() not in x.tolist()]
        x_val = x[val_index,:]
        y_val = y[val_index,:]
        return x_train,y_train,x_val,y_val

      #split data to train and test sets
      train_size = 1-test_size
      split_trainig_index = int(round(len(x) * train_size))
      x_train, y_train = x[:split_trainig_index], y[:split_trainig_index]
      x_test, y_test = x[split_trainig_index:], y[split_trainig_index:]

      return x_train,y_train,x_test,y_test
  

def preprocessing(dataframe):
    df = dataframe.copy()

    #drop rows if having null values
    if df.isna().values.any():
       df.dropna()

    # min-max normalization [0,1]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    x = df[['col1','col2']].to_numpy()
    y = df[['col3','col4']].to_numpy()

    return x,y

def hyperparameter_tuning(x,y):

    #hyperparameters grid for tuning
    learning_rates = [1e-1,1e-3,1e-5,1e-6]
    momentums = [8e-1,8e-3]

    final_lr = None
    final_momentum = None

    #data for hyperparameters tuning
    x_size = int(x.shape[0]/2)
    y_size = int(y.shape[0]/2)

    x_train = x[:x_size,:]
    y_train = y[:y_size,:]

    for lr in learning_rates:
        for momentum in momentums:

            print(f"Start tuning hyperparameters : learning_rate = {lr} and momentum = {momentum}")

            model =  Sequential()
            model.add(Layer(x_train.shape[1])) # Input layer
            model.add(Layer(4,activation="sigmoid")) # Hidden layer
            model.add(Layer(y_train.shape[1])) # Output layer

            sgd = SGD(learning_rate=lr,momentum=momentum)
            rmse = RootMeanSquaredError()

            model.compiler(optimizer=sgd, loss=rmse)

            fit = model.fit(x_train,y_train,epochs=100)
            
            if fit:
                print("No early stopping!")
                final_lr = lr
                final_momentum = momentum
                break

    return final_lr, final_momentum

def train_model(x_train,y_train):

    lr, momentum = hyperparameter_tuning(x_train,y_train) 

    model =  Sequential()
    model.add(Layer(x_train.shape[1])) # Input layer
    model.add(Layer(4,activation="sigmoid")) # Hidden layer
    model.add(Layer(y_train.shape[1])) # Output layer


    sgd = SGD(learning_rate=lr,momentum=momentum)

    rmse = RootMeanSquaredError()

    model.compiler(optimizer=sgd, loss=rmse)

    model.fit(x_train,y_train,epochs=100)

    save_path = 'model_saved.pth'
    model.save(save_path)

    return model

def test_model(model, x_test,y_test):

    evaluate = RootMeanSquaredError()

    y_pred = model.predict(x_test)

    rmse_evaluate = evaluate.rmse(y_test,y_pred)

    print(f"RMSE on test set is {rmse_evaluate}")

    return

def visualization(model):

    optimal_training_loss = min(model.training_losses)
    optimal_training_epoch = model.training_losses.index(min(model.training_losses))

    optimal_validation_loss = min(model.validation_losses)
    optimal_validation_epoch = model.validation_losses.index(min(model.validation_losses))

    print(f"Optimal point for Training set is Loss :{optimal_training_loss:.8f} at {optimal_training_epoch}")
    print(f"Optimal point for Validation set is Loss :{optimal_validation_loss:.8f} at {optimal_validation_epoch}")

    plt.scatter(optimal_training_epoch, optimal_training_loss, color='r', label=f'Optimal Point for Training set')
    plt.scatter(optimal_validation_epoch, optimal_validation_loss, color='b', label=f'Optimal Point for Validation set')

    plt.plot(model.training_losses, label='Training Loss')
    plt.plot(model.validation_losses, label='Validation Loss')

    plt.xlabel('Epochs (100)')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.legend()
    plt.savefig('Loss of >20K')
    plt.show()


if __name__ == "__main__":

    file_name = sys.argv[1]
    input_neuron = sys.argv[2]
    hidden_neuron = sys.argv[3]
    output_neuron = sys.argv[4]


    print(f"File name is {file_name}")

    print(f"Input neuron size : {input_neuron}, Hidden neuron size : {hidden_neuron}, Output neuron size : {output_neuron}")

    
    df = pd.read_csv(file_name, names=['col1','col2','col3','col4'])


    x,y = preprocessing(df)

    print(f"Features size : {len(x)}, Labels size : {len(y)}")

    validation_size = 0.15
    test_size = 0.15

    x_train,y_train,x_test,y_test = Model_Selection().train_test_split(x,y,test_size,False)

    model = train_model(x_train,y_train)

    test_model(model, x_test,y_test)

    visualization(visualization)