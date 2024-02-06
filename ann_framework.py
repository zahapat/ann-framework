import numpy as np
import multiprocessing
from visualizers import *
from model_wrapper import *
from dataset_fashion_mnist import *


# Do not upgrade seed
np.random.seed(0)

# Main Process
def main():

    # --------------------
    # -- START TRAINING --
    # --------------------

    # 1. Create a dataset
    # Note: Each 'list' in this larger list is a sample representing a feature set
    #       and are also referred to as feature set instances or observations.
    show_dataset = False

    # Regression problem dataset
    # X_train, y_train = sine_data()
    # X_valid, y_valid = sine_data()
    # sample_width = 1

    # Classification problem dataset; y (labels) must be a list of lists (inner list has one 0 or 1 output)
    # 1) Spiral data
    # number_of_classes = 5
    # X_train, y_train = spiral_data(samples=100, classes=number_of_classes)
    # X_valid, y_valid = spiral_data(samples=100, classes=number_of_classes)
    # number_of_classes = 10
    # sample_width = number_of_classes
    # batch_size = None

    # 2) Fashion MNIST
    # Create dataset, Preprocessing: scale all features, reshape matrices to single vectors, shuffle, create batches,
    X_train, y_train, X_valid, y_valid = dataset_mnist_create(display_image_number=None)

    sample_width = X_train.shape[1]
    number_of_classes = 10
    batch_size = 128

    realtime_monitor_process_pool = []
    xy_data_to_pipe_list = []
    destructor_pipes_list = []

    # Shared parameters among all processes
    training_epochs = 100
    sample_loss_after_steps = 10
    show_results_after_steps = 100

    # Process 1
    xy_data_to_pipe_list.append(multiprocessing.Pipe())  # [0]=TX, [1]=RX sender_xy_data_to_pipe
    destructor_pipes_list.append(multiprocessing.Pipe()) # [0]=TX, [1]=RX receiver_destructor_pipe
    sample_width = sample_width
    number_of_classes = number_of_classes
    training_data = X_train, y_train
    validation_data = X_valid, y_valid
    test_data = validation_data
    nn_distribution_init = "normal"
    layer1_neurons_count = 128
    additional_hidden_layers = 1
    optimizer_function = "adam"
    optimizer_function_parameters = {
        # Default Values (Adam)
        "learning_rate": 0.001,
        # "exponential_decay": 0.,
        "epsilon": 1e-7,
        "beta_1": 0.9,
        "beta_2": 0.999,

        # User-defined parameters
        # "learning_rate": 0.05,
        "exponential_decay": 1e-4,
        # "epsilon": 1e-7,
        # "beta_1": 0.9,
        # "beta_2": 0.999
    }
    output_activation_function = "softmax"
    dropout_rate = 0.0
    sender_xy_data_to_pipe = xy_data_to_pipe_list[-1][0]
    receiver_destructor_pipe = destructor_pipes_list[-1][1]
    this_process_id = len(xy_data_to_pipe_list)-1

    realtime_monitor_process_pool.append(multiprocessing.Process(
        target=model_wrapper,
        args=(
            # Do not touch:
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
            this_process_id,)
    ))


    # Process 2
    # xy_data_to_pipe_list.append(multiprocessing.Pipe())  # [0]=TX, [1]=RX
    # destructor_pipes_list.append(multiprocessing.Pipe()) # [0]=TX, [1]=RX
    # input_dimensionality = 1
    # dataset_X = X
    # dataset_y = y
    # dataset_test_X = valid_X
    # dataset_test_y = valid_y
    # dataset_number_of_classes = number_of_classes
    # nn_distribution_init = "normal"
    # layer1_neurons_count = 28
    # # layer1_neurons_count = 24
    # dropout_rate = 0.1
    # optimizer = "SGD"
    # sender_xy_data_to_pipe = xy_data_to_pipe_list[-1][0]
    # receiver_destructor_pipe = destructor_pipes_list[-1][1]
    # this_process_id = (len(xy_data_to_pipe_list)-1)
    # realtime_monitor_process_pool.append(multiprocessing.Process(
    #     target=model_wrapper,
    #     args=(
    #         input_dimensionality,
    #         dataset_X,
    #         dataset_y,
    #         dataset_test_X,
    #         dataset_test_y,
    #         dataset_number_of_classes,
    #         nn_distribution_init,
    #         layer1_neurons_count,
    #         dropout_rate,
    #         training_epochs,
    #         downsample_loss,
    #         test_after_epochs,
    #         optimizer,
    #         sender_xy_data_to_pipe,
    #         receiver_destructor_pipe,
    #         this_process_id,)
    # ))




    # ----------------------
    # -- START MONITORING --
    # ----------------------
    # Array of Forked Jobs
    [realtime_monitor_process_pool[i].start() for i in range(len(realtime_monitor_process_pool))]

    # Main Core Job
    receiver_xy_data_pipe_list = [xy_data_to_pipe_list[i][1] for i in range(len(xy_data_to_pipe_list))]
    sender_destructor_pipes_list = [destructor_pipes_list[i][0] for i in range(len(destructor_pipes_list))]
    thread_realtime_monitor(receiver_xy_data_pipe_list, sender_destructor_pipes_list)

    # Join Array of Forked Jobs
    [realtime_monitor_process_pool[i].join() for i in range(len(realtime_monitor_process_pool))]


# This is the main file of the script
if __name__ == '__main__':
    main()