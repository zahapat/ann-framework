from model import *
from optimizers import *

# Model Wrapper - Infers classificatin or regression problem
def model_wrapper(
        sample_width,
        number_of_classes,
        training_data,
        validation_data,
        test_data,
        nn_distribution_init,
        layer1_neurons_count,
        additional_hidden_layers,
        optimizer_function,
        optimizer_function_parameters,
        output_activation_function,
        dropout_rate,
        training_epochs,
        batch_size,
        sample_loss_after_steps,
        show_results_after_steps,
        sender_xy_data_to_pipe,
        receiver_destructor_pipe,
        this_process_id,):

    # Start timer
    time_start = time.time()

    # Create a Model and Connect neurons
    model = Model(receiver_destructor_pipe, sender_xy_data_to_pipe,
                  this_process_id)
    
    if number_of_classes == 1: # This should be dimensionality
        # Regression Problems (Sine data, ...)
        # Input Layer
        model.add(Layer_Dense(1, layer1_neurons_count))
        model.add(Activation_ReLU())
        if dropout_rate > 0: 
            model.add(Layer_Dropout(dropout_rate))

        # Hidden Layers
        if additional_hidden_layers > 0:
            for hidden_layers in range(additional_hidden_layers):
                model.add(Layer_Dense(layer1_neurons_count, layer1_neurons_count))
                model.add(Activation_ReLU())
                if dropout_rate > 0: 
                    model.add(Layer_Dropout(dropout_rate))

        # Output Layers
        model.add(Layer_Dense(layer1_neurons_count, 1))
        model.add(Activation_Linear())
        loss_function = Loss_MeanSquaredError()
        accuracy_function = Accuracy_Regression()
    else:
        # Classification Problems (Spiral data, Fashion MNIST, ...)
        # Input Layer
        model.add(Layer_Dense(sample_width, layer1_neurons_count,
                              l2_weight_regularizer=5e-4,
                              l2_bias_regularizer=5e-4))
        model.add(Activation_ReLU())
        if dropout_rate > 0: 
            model.add(Layer_Dropout(dropout_rate))

        # Hidden Layers
        if additional_hidden_layers > 0:
            for new_hidden_layer in range(additional_hidden_layers):
                model.add(Layer_Dense(layer1_neurons_count, layer1_neurons_count))
                model.add(Activation_ReLU())
                if dropout_rate > 0: 
                    model.add(Layer_Dropout(dropout_rate))

        # Output Layers
        if output_activation_function == "sigmoid":
            model.add(Layer_Dense(layer1_neurons_count, 1))
            model.add(Activation_Sigmoid())
            loss_function = Loss_BinaryCrossentropy()
            training_data = list(training_data)
            validation_data = list(validation_data)
            training_data[1] = training_data[1].reshape(-1, 1)
            validation_data[1] = validation_data[1].reshape(-1, 1)
            training_data = tuple(training_data)
            validation_data = tuple(validation_data)

        elif output_activation_function == "softmax":
            model.add(Layer_Dense(layer1_neurons_count, number_of_classes))
            model.add(Activation_Softmax())
            loss_function = Loss_CategoricalCrossEntropy()

        accuracy_function = Accuracy_Categorical()

    # Set Desired Loss, Optimizer and Accuracy function
    if optimizer_function == "sgd":
        optimizer_function = Optimizer_SGD(**optimizer_function_parameters)
    elif optimizer_function == "adagrad":
        optimizer_function = Optimizer_AdaGrad(**optimizer_function_parameters)
    elif optimizer_function == "rmspropag":
        optimizer_function = Optimizer_RMSPropag(**optimizer_function_parameters)
    elif optimizer_function == "adam":
        optimizer_function = Optimizer_Adam(**optimizer_function_parameters)

    model.set(Loss=loss_function,
              optimizer=optimizer_function,
              accuracy=accuracy_function)

    # Connect all layers
    model.finalize()

    # Train the network
    model.train(training_data,
                epochs=training_epochs,
                batch_size=batch_size,
                sample_loss_after_steps=sample_loss_after_steps, 
                show_results_after_steps=show_results_after_steps,
                validation_data=validation_data)
    
    # Evaluate on Test data (not-yet-seen out-of-sample data)
    model.evaluate(*test_data, "test_data")

    # Evaluate on Training data (get the final accuracy and loss)
    model.evaluate(*training_data, "training_data")

    model.export_weights_biases()

    print(f'PY: Learning has finished: process {this_process_id} took {time.time()-time_start} sec to complete. Close the window to exit.')