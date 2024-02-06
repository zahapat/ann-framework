import numpy as np
from print import Print

# An artificial input layer
# Perform Forward Pass specifically for the first layer in order to perform initialization uniformly
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


# A fully interconnected layer
class Layer_Dense(Print):
    def __init__(self, inputs_count, neurons_count, \
                 l1_weight_regularizer=0, l1_bias_regularizer=0, l2_weight_regularizer=0, l2_bias_regularizer=0, \
                 verbose=False, distribution='normal', init_weights_value=0.33333):

        # Create initial weights - ideally values within boundaries -1 and +1
        if distribution == 'normal':
            self.weights = 0.01 * np.random.randn(inputs_count, neurons_count)
        elif distribution == 'uniform':
            self.weights = np.random.uniform(
                low=-1,
                high=1,
                size=(inputs_count, neurons_count)
            )
        elif distribution == 'degenerate':
            self.weights = init_weights_value * np.ones(shape=(inputs_count, neurons_count))

        # Biases will be defaulted to 0
        self.biases = np.zeros(shape=(1, neurons_count))

        # L1 and L2 Regularization
        self.l1_weight_regularizer = l1_weight_regularizer
        self.l1_bias_regularizer = l1_bias_regularizer
        self.l2_weight_regularizer = l2_weight_regularizer
        self.l2_bias_regularizer = l2_bias_regularizer

        if verbose:
            self.print_verbose("self.weights", self.weights)
            self.print_verbose("self.biases", self.biases)
            self.print_verbose("self.l1_weight_regularizer", self.l1_weight_regularizer)
            self.print_verbose("self.l1_bias_regularizer", self.l1_bias_regularizer)
            self.print_verbose("self.l2_weight_regularizer", self.l2_weight_regularizer)
            self.print_verbose("self.l2_bias_regularizer", self.l2_bias_regularizer)

    # Perform dot product on input batch
    def forward(self, inputs, training, verbose=False):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if verbose:
            self.print_verbose("self.output", self.output)

    # Perform Backpropagation, deri_activation=drelu
    def backward(self, deri_activation, verbose=False):
        # Gradient on weights and biases
        self.dweights = np.dot(self.inputs.T, deri_activation)
        self.dbiases = np.sum(deri_activation, axis=0, keepdims=True)

        # Gradient on backpropagated values
        self.dinputs = np.dot(deri_activation, self.weights.T)

        # Gradient on Regularizations
        # L1
        if self.l1_weight_regularizer > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1_weight_regularizer * dL1
        if self.l1_bias_regularizer > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1_bias_regularizer * dL1
        # L2
        if self.l2_weight_regularizer > 0:
            self.dweights += 2 * self.l2_weight_regularizer * self.weights
        if self.l2_bias_regularizer > 0:
            self.dbiases += 2 * self.l2_bias_regularizer * self.biases

        if verbose:
            self.print_verbose("self.dweights", self.dweights)
            self.print_verbose("self.dbiases", self.dbiases)
            self.print_verbose("self.dinputs", self.dinputs)
    
    # Get actual weight and biases
    def get_weights_biases(self):
        return self.weights, self.biases

    # Set new weights and biases
    def set_weights_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout layer: Regularization that disables neurons to helps generalize; prevents co-adoption, overfitting
#                Dr_i = z_i/(1-q) if r_i=1
#                Dr_i = 0         if r_i=0
#         d/dz_i Dr_i = 1/(1-q)   if r_i=1
#         d/dz_i Dr_i = 0         if r_i=0
# i ... index of the input, Dr ... dropout, 
# z ... neuron's output, q ... dropout rate,
# r ... random draw from Bernoulli Distr
class Layer_Dropout(Print):
    def __init__(self, dropout_rate, verbose=False):
        # Calculate the 1-q used in all equations below
        self.success_rate = 1-dropout_rate
        self.apply_dropout = True
        if dropout_rate <= 0: self.apply_dropout = False

        if verbose:
            self.print_verbose("self.success_rate", self.success_rate)

    def forward(self, inputs, training, verbose=False):
        # Leave early if not training - Dropout layer is only used in training
        if not training:
            # Do not apply any mask for dropout
            self.output = inputs.copy()
            return

        # Apply dropout layer only if dropout rate is above 0, else omit this block
        if self.apply_dropout == True:
            # Outputs from the Dense_Layer are inputs to the Dropout_Layer
            self.inputs = inputs

            # Mask created using Bernoulli distribution to switch off neurons in a layer randomly with respect to dropout rate
            # Mask values need to be scaled to handle the difference in execution with drouput layer during
            # training and without dropout during prediction or validation to mimick that all neurons output some values
            self.binary_mask = np.random.binomial(
                                1, self.success_rate, size=inputs.shape) / self.success_rate

            # Apply mask for dropout
            self.output = inputs * self.binary_mask

            if verbose:
                self.print_verbose("self.inputs", self.inputs)
                self.print_verbose("self.binary_mask", self.binary_mask)
                self.print_verbose("self.output", self.output)
        else:
            # Do not apply mask
            self.output = inputs

            if verbose:
                self.print_verbose("self.inputs", self.inputs)


    def backward(self, dvalues, verbose=False):
        # Apply dropout layer only if dropout rate is above 0, else omit this block
        if self.apply_dropout == True:
            # Calculate gradients using Chain rule
            self.dinputs = dvalues * self.binary_mask

            if verbose:
                self.print_verbose("self.dinputs", self.dinputs)
        else:
            # Do not calculate gradients
            self.dinputs = dvalues

            if verbose:
                self.print_verbose("self.dinputs", self.dinputs)