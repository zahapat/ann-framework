import numpy as np
from print import Print
from activations import Activation_Softmax


# Inherited by every Loss class
class Loss(Print):

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self, verbose=False):

        # Set 0 as the default value for calculation
        regularization_loss = 0

        # Regularization loss for all layers
        for layer in self.trainable_layers:

            #L1 Regularization
            if layer.l1_weight_regularizer > 0:
                regularization_loss += layer.l1_weight_regularizer * np.sum(np.abs(layer.weights))
            if layer.l1_bias_regularizer > 0:
                regularization_loss += layer.l1_bias_regularizer * np.sum(np.abs(layer.biases))
            
            #L2 Regularization
            if layer.l2_weight_regularizer > 0:
                regularization_loss += layer.l2_weight_regularizer * np.sum(layer.weights*layer.weights)
            if layer.l2_bias_regularizer > 0:
                regularization_loss += layer.l2_bias_regularizer * np.sum(layer.biases*layer.biases)
        
        if verbose:
            self.print_verbose("layer.l1_weight_regularizer", layer.l1_weight_regularizer)
            self.print_verbose("layer.l1_bias_regularizer", layer.l1_bias_regularizer)
            self.print_verbose("layer.l2_weight_regularizer", layer.l2_weight_regularizer)
            self.print_verbose("layer.l2_bias_regularizer", layer.l2_bias_regularizer)
            self.print_verbose("regularization_loss", regularization_loss)

        return regularization_loss

    def calculate(self, output_activation, y_target_values, *, return_regularization_loss=False, verbose=False):
        # Sample losses
        sample_losses = self.forward(
            output_activation, 
            y_target_values,
            verbose=verbose)

        # Calculate data loss as mean value
        data_loss = np.mean(sample_losses)

        # Accumulate sample losses if we are sending multiple batches per epoch
        self.all_sample_losses_batches_accumulated += np.sum(sample_losses) # sample losses in 1 batch
        self.all_sample_losses_count += len(sample_losses) # number of sample losses in 1 batch

        if verbose:
            self.print_verbose("sample_losses", sample_losses)
            self.print_verbose("data_loss", data_loss)

        # Do not output regularization loss - return only data loss (for classification problems)
        # Return regularization losses and data loss for all layers (for regression problems)
        if not return_regularization_loss:
            return data_loss
        return data_loss, self.regularization_loss()

    # Accumulated loss for accumulated batches of inputs
    def calculate_accumulated(self, *, return_regularization_loss=False):
        # Calculate mean loss
        data_loss = self.all_sample_losses_batches_accumulated / self.all_sample_losses_count

        # Do not output regularization loss - return only data loss (for classification problems)
        # Return regularization losses and data loss for all layers (for regression problems)
        if not return_regularization_loss:
            return data_loss
        return data_loss, self.regularization_loss()

    # Reset accumulated losses
    def reset_calculate_accumulated(self):
        self.all_sample_losses_batches_accumulated = 0
        self.all_sample_losses_count = 0


class Loss_CategoricalCrossEntropy(Loss, Print):
    def forward(self, y_predicted_values, intended_target_values, verbose=False):
        y_predicted_values_len = len(y_predicted_values)

        # Example 1 : Class selection 1d shape:                 [1    ,0    ,1    ,1    ]
        if len(intended_target_values.shape) == 1:
            correct_confidences = y_predicted_values[
                range(y_predicted_values_len),
                intended_target_values
            ]
        # Example 2 : One-hot Encoded Class selection 2d shape: [[1,0],[0,1],[1,0],[1,0]]
        elif len(intended_target_values.shape) == 2:
            correct_confidences = np.sum(
                y_predicted_values*intended_target_values,
                axis=1
            )

        # Calculate losses
        # Prevent divide by zero encountered in log: infinity problem
        correct_confidences_clipped = np.clip(correct_confidences, 1e-7, 1-1e-7)
        negative_log_likelihoods = -np.log(correct_confidences_clipped)

        if verbose:
            self.print_verbose("correct_confidences_clipped", correct_confidences_clipped)
            self.print_verbose("negative_log_likelihoods", negative_log_likelihoods)

        return negative_log_likelihoods

    def backward(self, predicted_true_values, intended_target_values, verbose=False):
        samples_count_samplewise = len(predicted_true_values)
        lables_count_batchwise = len(predicted_true_values[0])

        if len(intended_target_values.shape) == 1:
            intended_target_values = np.eye(lables_count_batchwise)[intended_target_values]

        # Calculate gradient of the Categorical Cross-Entropy
        self.dinputs = -intended_target_values / predicted_true_values

        # Normalize gradient
        self.dinputs = self.dinputs / samples_count_samplewise

        if verbose:
            self.print_verbose("samples_count_samplewise", samples_count_samplewise)
            self.print_verbose("lables_count_batchwise", lables_count_batchwise)
            self.print_verbose("intended_target_values", intended_target_values)
            self.print_verbose("self.dinputs", self.dinputs)



class Loss_BinaryCrossentropy(Loss, Print):
    def forward(self, y_pred, y_true, verbose=False):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        sample_losses = - (  y_true  * np.log(  y_pred_clipped)
                        + (1-y_true) * np.log(1-y_pred_clipped))
        if verbose:
            self.print_verbose("y_pred_clipped", y_pred_clipped)
            self.print_verbose("sample_losses", sample_losses)

        return np.mean(sample_losses, axis=-1) # Sample Losses Normalized

    def backward(self, dvalues, y_true, verbose=False):
        samples_count = len(dvalues)
        outputs_count = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        self.dinputs = - (  y_true  /    clipped_dvalues
                       - (1-y_true) / (1-clipped_dvalues))
        if verbose:
            self.print_verbose("samples_count", samples_count)
            self.print_verbose("outputs_count", outputs_count)
            self.print_verbose("clipped_dvalues", clipped_dvalues)
            self.print_verbose("self.dinputs", self.dinputs)


# Use this loss function for Regression problems
# Take the squared difference between predicted and true values of single outputs, average those
# - in case you cannot work with classification lables
class Loss_MeanSquaredError(Loss, Print):
    def forward(self, y_predicted_values, y_true_target_values, verbose=False):
        # Calculate the Mean Squared Error (loss)
        sample_losses = np.mean((y_true_target_values - y_predicted_values)**2, axis=-1)

        if verbose:
            pass

        return sample_losses

    def backward(self, dvalues, y_true_target_values, verbose=False):
        # Get the dimensions of the dvalues input variable
        samples_count = len(dvalues)
        outputs_count = len(dvalues[0])

        # Calculate the gradient of the Mean Squared Error (loss)
        self.dinputs = -2 * (y_true_target_values - dvalues) / outputs_count

        # Normalize values
        self.dinputs = self.dinputs / samples_count

        if verbose:
            pass

# Used for Regression problems
# Take the absolute difference between predicted and true values of single outputs, average those
class Loss_MeanAbsoluteError(Loss, Print):
    def forward(self, y_predicted_values, y_true_target_values, verbose=False):
        # Calculate the Mean Absolute Error (loss)
        sample_losses = np.mean(np.abs(y_true_target_values - y_predicted_values), axis=-1)

        if verbose:
            pass

        return sample_losses
    
    def backward (self, dvalues, y_true_target_values, verbose=False):
        # Get the dimensions of the dvalues input variable
        samples_count = len(dvalues)
        outputs_count = len(dvalues[0])

        # Calculate the gradient of the Mean Absolute Error (loss)
        self.dinputs = np.sign(y_true_target_values - dvalues) / outputs_count

        # Normalize values
        self.dinputs = self.dinputs / samples_count

        if verbose:
            pass



# Faster Softmax classifier - combined Softmax activation
# and cross-entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy(Print):
    def __init__(self):
        self.activation_softmax = Activation_Softmax()
        self.loss_categoricalcrossentropy = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true_target_values, verbose=False):
        self.activation_softmax.forward(inputs)
        self.output = self.activation_softmax.output
        if verbose:
            self.print_verbose("self.output", self.output)
            self.print_verbose("self.loss_categoricalcrossentropy.calculate(self.output, y_true_target_values)", self.loss_categoricalcrossentropy.calculate(self.output, y_true_target_values))
        return self.loss_categoricalcrossentropy.calculate(self.output, y_true_target_values)

    def backward(self, dvalues_onehot, y_true_target_values, verbose=False):
        samples_count = len(dvalues_onehot)

        if len(y_true_target_values.shape) == 2:
            y_true_target_values = np.argmax(y_true_target_values, axis=1)

        self.dinputs = dvalues_onehot.copy()

        # Calculate gradient
        self.dinputs[range(samples_count), y_true_target_values] -= 1

        # Normalize Gradient
        self.dinputs = self.dinputs / samples_count

        if verbose:
            self.print_verbose("samples_count", samples_count)
            self.print_verbose("y_true_target_values", y_true_target_values)
            self.print_verbose("self.dinputs", self.dinputs)