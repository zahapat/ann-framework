import os
import time
import copy
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from print import Print
from losses import *
from layers import *
from accuracies import *
from activations import *

# The global class Model: it needs to be explicitly defined which layer has no previous layers (not a hidden layer)
class Model:
    # Initialize the Model with empty lists and configuration data
    def __init__(self, 
                 receiver_destructor_pipe, sender_xy_data_to_pipe,
                 this_process_id=0):
        self.model_id = this_process_id
        self.receiver_destructor_pipe = receiver_destructor_pipe
        self.sender_xy_data_to_pipe = sender_xy_data_to_pipe
        self.all_layers = []

        # Initialize Softmax Classifier Detection
        self.softmax_classifier_output = None

    # Add a new layer
    def add(self, layer):
        self.all_layers.append(layer)

    # Set Optimizer and Loss class to be used in the model
    def set(self, *, Loss, optimizer, accuracy):
        # If we only load weights biases (don't train) we don't need an optimizer
        # thus, loss, accuracy, and optimizer can be ommitted
        if Loss is not None:
            self.loss = Loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None: 
            self.accuracy = accuracy

    def find_trainable_layers(self):
        self.trainable_layers = []
        for i in range(self.layers_count):
            if hasattr(self.all_layers[i], 'weights'):
                self.trainable_layers.append(self.all_layers[i])

    # Set previous and next layer properties to interconnect the entire neural network
    def finalize(self):
        # The first (not hidden), artificial layer
        self.input_layer = Layer_Input()
        self.layers_count = len(self.all_layers)

        # Set object (layers) references for the forward and backward methods
        for i in range(self.layers_count):
            # Layer 0 acts like a hidden layer as it is referenced to another (artificial) previous layer
            if i == 0:
                self.all_layers[i].prev = self.input_layer
                self.all_layers[i].next = self.all_layers[i+1]
            elif i < self.layers_count-1:
                self.all_layers[i].prev = self.all_layers[i-1]
                self.all_layers[i].next = self.all_layers[i+1]
            else:
                self.all_layers[i].prev = self.all_layers[i-1]
                self.all_layers[i].next = self.loss

                # Link / define the layer before loss calculation as the output layer activation
                # Such as Softmax, Linear, ...
                self.output_layer_activation = self.all_layers[i]
        
        # Find and update trainable layers to calculate loss in all layers
        self.find_trainable_layers()

        # If we only load weights biases (don't train) we don't need an optimizer
        # thus, loss, accuracy, and optimizer can be ommitted
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # Implement the combined Softmax and CategoricalCrossEntropy class
        # if both Softmax and CategoricalCrossEntropy classes are present
        softmax_present = isinstance(self.all_layers[-1], Activation_Softmax)
        categoricalcrossentropy_present = isinstance(self.loss, Loss_CategoricalCrossEntropy)
        if softmax_present and categoricalcrossentropy_present:
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    # Unified forward method for all layers using object references (see finalize() method)
    def forward(self, X, training=False):
        # The Input Layer has its own forward method, and is the first referenced object
        # its input is training data, outputs also training data
        self.input_layer.forward(X, training)

        # The layer 0 is the first layer which acts like a hidden layer
        # and takes training data from input_layer (which is the .prev reference).
        for layer in self.all_layers:
            layer.forward(layer.prev.output, training)

        # The layer variable is now last object in the all_layers list
        return layer.output
    
    # Define figure window for live plot, enable interactive mode, initialize plots
    def init_live_plot(self, initial_data):
        # Unpack validation data
        X, y = initial_data

        # Initialize the window and figure with data
        try:
            plt.ion()
            self.figure_window = plt.figure()
            all_axes = self.figure_window.add_subplot(1, 1, 1) # subplots on 1x1 grid, item 1
            self.plot_line_test, = all_axes.plot(X, y)
            self.plot_line_predictions, = all_axes.plot(X, np.zeros_like(X))
            display = True
        except:
            print(f'PY: Error Handler in init_live_plot: Displaying progress is not supported.')
            plt.ioff()
            plt.close(self.figure_window)
            display = False

        return display

    
    # Evaluate = Validate (using) or Test (usinng out-of-sample data) the neural network
    def evaluate(self, X_eval, y_eval, description="unspecified_dataset", *, batch_size=None, display=False):

        # Set validation steps to default 1 in case batch size is not set
        if (X_eval is not None) and (y_eval is not None):
            validation_steps = 1
        else:
            print(f'PY: Warning: No data available for running evaluate(). Skip running this method.')
            return 0

        # Calculate number of steps if batch size is set, add one if not divisible
        if batch_size is not None:
            # Round down integer division, if reminder, add one step to process all samples
            validation_steps = len(X_eval) // batch_size
            if validation_steps * batch_size < len(X_eval):
                validation_steps += 1

        # Reset accumulated accuracy for accumulated loss and accuracy over batches
        self.loss.reset_calculate_accumulated()
        self.accuracy.reset_calculate_accumulated()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_eval
                batch_y = y_eval
            else:
                batch_X = X_eval[(step)*batch_size : (step+1)*batch_size]
                batch_y = y_eval[(step)*batch_size : (step+1)*batch_size]

            # Perform forward pass, get the output of the forward pass
            output = self.forward(batch_X)

            # Calculate losses: Regularization loss is not used during validation
            loss = self.loss.calculate(output, batch_y)

            # Get Predictions, and calculate Accuracy based on these predictions
            # Every output activation function must have a method to calculate predictions
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
        
        # Get accumulated loss and accuracies across all batches (steps)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Report 
        print(f'PY: Model {self.model_id}: '+
            f'e/s/a: {self.epoch}/{self.step}/{self.all_training_steps}, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f} ' +
            f'({str(description)})'
        )

        # Display testing data and predictions at the current epoch
        if display:
            self.plot_line_predictions.set_ydata(output)
            self.figure_window.canvas.flush_events()

        return None
    
    def check_close_evnet(self):
        # If break requested from realtime plot, send status done to realtime plot as a handshake
        if self.receiver_destructor_pipe.poll():
            if self.receiver_destructor_pipe.recv():
                print(f'PY: StopProcess requested. Training has ended by user.')
                self.sender_xy_data_to_pipe.send([None, None, 'StopProcess'])
                return True

    # Train the neural network
    def train(self, training_data, *, epochs=1, batch_size=None, sample_loss_after_steps=1, show_results_after_steps=1,
              validation_data=None):

        # Initialize training: training and validation steps, live plot, accuracy precision, training data
        X, y = training_data
        display = self.init_live_plot(validation_data)
        training_steps = 1
        self.all_training_steps = 0
        self.accuracy.set_precision(y)

        # Calculate number of steps if batch size is set, add one if not divisible
        if batch_size is not None:
            # Round down integer division, if reminder, add one step to process all samples
            training_steps = len(X) // batch_size
            if training_steps * batch_size < len(X):
                training_steps += 1

        # Train over defined number of epochs
        for epoch in range(epochs+1):

            # Initialize timer for this epoch
            timer_start = time.time()
            self.epoch = epoch
            print(f'PY: --------------------')
            print(f'PY: Model {self.model_id}: epoch: {self.epoch}')

            # Reset accumulated accuracy for accumulated loss and accuracy over batches
            # Make epoch visible to other methods in this class
            self.loss.reset_calculate_accumulated()
            self.accuracy.reset_calculate_accumulated()

            for step in range(training_steps):
                self.step = step

                # If break requested from realtime plot, send status done to realtime plot as a handshake
                if self.check_close_evnet() == True: return 0

                if batch_size is None:
                    # One batch is the entire dataset
                    batch_X = X
                    batch_y = y
                else:
                    # Take a different batch of the dataset of defined size on every step
                    batch_X = X[(step)*batch_size : (step+1)*batch_size]
                    batch_y = y[(step)*batch_size : (step+1)*batch_size]

                # Perform forward pass, get the output of the forward pass, 
                # set training=True for Dropout Layer as it is used only during training
                output = self.forward(batch_X, training=True)

                # Calculate losses
                data_loss, regularization_loss = self.loss.calculate(output, batch_y,
                                                    return_regularization_loss=True)
                loss = data_loss + regularization_loss

                # Get Predictions, and calculate Accuracy based on these predictions
                # Every output activation function must have a method to calculate predictions
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform the backward pass using the output of the forward pass
                self.backward(output, batch_y)

                # Update weights and biases
                self.optimizer.update_learning_rate()
                for layer in self.trainable_layers:
                    self.optimizer.update_weights_biases(layer)
                self.optimizer.increment_epoch_counter()

                # Each downsample_loss-th epoch, send current results to pipe
                if not self.all_training_steps % sample_loss_after_steps:
                    self.sender_xy_data_to_pipe.send([self.all_training_steps, loss, None])

                if not self.all_training_steps % show_results_after_steps:
                    # Show intermediate results
                    print(f'PY: Model {self.model_id}: '+
                        f'e/s/a: {self.epoch}/{self.step}/{self.all_training_steps}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} ' +
                        f'(data: {data_loss:.3f}, ' +
                        f'reg: {regularization_loss:.3f}), ' +
                        f'lrate: {self.optimizer.learning_rate_current:.4f}')

                # Increment the step counter
                self.all_training_steps += 1

            # Calculate losses across all (accumulated) steps
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(
                                                return_regularization_loss=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss

            # Calculate Accuracy based on these predictions (now not needed, calculated during steps)
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # Show intermediate results
            print(f'PY: Model {self.model_id}: '+
                f'e/s/a: {self.epoch}/{self.step}/{self.all_training_steps}, ' +
                f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f} ' +
                f'(data: {epoch_data_loss:.3f}, ' +
                f'reg: {epoch_regularization_loss:.3f}), ' +
                f'lrate: {self.optimizer.learning_rate_current:.4f}')

            # Validate during training if validation adata available, display progress
            if validation_data:
                self.evaluate(*validation_data, "validation_data", batch_size=batch_size, display=display)
            
            # Print timer
            print(f'PY: Model {self.model_id}: epoch: {self.epoch} timer: {time.time()-timer_start:.2f} (incl. validation_data)')

            # If break requested from realtime plot, send status done to realtime plot as a handshake
            if self.check_close_evnet() == True: return 0

        # Report done and send status done to realtime plot to close the window
        self.sender_xy_data_to_pipe.send([None, None, 'StopProcess'])

        # Keep the plot window open until closed manually (only possible in non-interactive mode)
        if display:
            plt.ioff()
            plt.show()

    def backward(self, output, y):
        # Asymmetrical number of layers for forward and backpropagation: 
        # due to the presence of the combined layer out of two layers
        # Modify the layer graph if softmax classifier is present (detected in finalize stage)
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            # Skip calling backward on the last layer (Softmax Activation) 
            # and use the dvalues of the combined Softmax and Cat. Cross Entr.
            self.all_layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Apply backpropagation for the remaining layers 
            # Note: python excludes the -1=last index
            for layer in reversed(self.all_layers[:-1]):
                layer.backward(layer.next.dinputs)

            return None

        # Perform the Chain rule backward propagation in a normal way
        # Symmetrical number of layers for forward and backpropagation
        self.loss.backward(output, y)
        for layer in reversed(self.all_layers):
            layer.backward(layer.next.dinputs)
    
    # Get weights and biases from trainable dense layers
    def get_weights_biases(self):

        # Iterable over all trainable layers, append weights and biases
        trainable_layers_weights_and_biases = []
        for trainable_layer in self.trainable_layers:
            trainable_layers_weights_and_biases.append(
                trainable_layer.get_weights_biases())

        # Return a list
        return trainable_layers_weights_and_biases

    # Set weights and biases
    def set_weights_biases(self, new_weights_biases):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for weights_biases, layer in zip(new_weights_biases, self.trainable_layers):
            layer.set_weights_biases(*weights_biases)
    
    # Export weights and biases to a file
    def export_weights_biases(self, *,
                              output_directory='params',
                              output_file_name=None):
        
        # Set default file name if None
        if output_file_name is None:
            output_file_name = f'model_{self.model_id}.params'

        # Correct path for the current OS
        output_directory = pathlib.Path(output_directory)

        # Create directory if path does not exist
        if not os.path.exists(f"{output_directory}"):
            os.mkdir(f"{output_directory}")
            print(f"PY: Created new directory: {output_directory}")

        # File handler: binary-write mode
        output_file_path = os.path.join(output_directory, output_file_name)
        with open(output_file_path, 'wb') as file:
            pickle.dump(self.get_weights_biases(), file)
        
        print(f'PY: Exported parameters into file: {output_file_path}')
    
    # Loads the weights and updates a model instance with them
    def import_weights_biases(self, *,
                              output_directory='params',
                              output_file_name=None):

        # Set default file name if None
        if output_file_name is None:
            output_file_name = f'model_{self.model_id}.params'

        # Correct path for the current OS
        output_directory = pathlib.Path(output_directory)

        # Check if directory exists
        if not os.path.exists(f"{output_directory}"):
            print(f"PY: Warning: Directory: {output_directory} does not exist. Nothing will be imported. Return.")
            return

        # File handler: binary-read mode
        output_file_path = os.path.join(output_directory, output_file_name)      
        with open(output_file_path, 'rb') as file:
            self.set_weights_biases(pickle.load(file))

        print(f'PY: Imported parameters from file: {output_file_path}')

    # Model deep copy: Export the entire image of the Model class with its state and sub-classes
    def export_model(self, *,
                     output_directory='models',
                     output_file_name=None):

        # Set default file name if None
        if output_file_name is None:
            output_file_name = f'model_{self.model_id}.model'

        # Correct path for the current OS
        output_directory = pathlib.Path(output_directory)

        # Create directory if path does not exist
        if not os.path.exists(f"{output_directory}"):
            os.mkdir(f"{output_directory}")
            print(f"PY: Created new directory: {output_directory}")

        # Make an image of this model instance
        model_checkpoint = copy.deepcopy(self)

        # Remove biases from the last training
        # Remove accumulated values inside the model image
        model_checkpoint.loss.reset_calculate_accumulated()
        model_checkpoint.accuracy.reset_calculate_accumulated()

        # Remove data from the input layer and gradients from the loss object: 
        # Set error message to None as its default output is an Error
        model_checkpoint.input_layer.__dict__.pop('output', None)
        model_checkpoint.loss.__dict__.pop('dinputs', None)

        # Remove inputs, output, dinputs, dweights, dbiases from each layer
        for layer in model_checkpoint.all_layers:
            for property in ['inputs', 'output', 'dinputs', 
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # File handler: binary-write mode
        output_file_path = os.path.join(output_directory, output_file_name)
        with open(output_file_path, 'wb') as file:
            pickle.dump(model_checkpoint, file)
        
        print(f'PY: Exported model into file: {output_file_path}')
    
    # Loads a model checkpoint before this object instance exists (using the staticmethod decorator)
    # Thus, having an instance of any model class created beforehand is not required
    @staticmethod
    def import_model(self, *,
                    output_directory='models',
                    output_file_name=None):

        # Set default file name if None
        if output_file_name is None:
            output_file_name = f'model_{self.model_id}.model'

        # Correct path for the current OS
        output_directory = pathlib.Path(output_directory)

        # Check if directory exists
        if not os.path.exists(f"{output_directory}"):
            print(f"PY: Warning: Directory: {output_directory} does not exist. Nothing will be imported. Return.")
            return

        # File handler: binary-read mode
        output_file_path = os.path.join(output_directory, output_file_name)      
        with open(output_file_path, 'rb') as file:
            model_checkpoint = pickle.load(file)
        
        # Return model checkpoint
        return model_checkpoint
    

    # Make predictions
    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            # Round down integer division, if reminder, add one step to process all samples
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        # Iterate over prediction steps based on batch size, otherwise use the entire dataset
        predictions = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[(step)*batch_size : (step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            predictions.append(batch_output)

        # Stack arrays in sequence vertically and return predictions
        return np.vstack(predictions)