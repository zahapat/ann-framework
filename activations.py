# Activation Functions
# - If we use only a linear activation function -> Output will be linear
#    -> With linear functions, only linear functions can be fitted with a high precision
#        -> We won't be able to solve non-linear problems
# - Activation functions, such as Activation_ReLU, are able to fit/approximate non-linear functions
#    -> They start as linear approximations but later the approximation will fit the target nonlinear function
# - Why does that work?
#    -> The core of neural networks resides in all the non-linear activation functions 
#       present in the entire network working together
#    -> Activation_ReLU is almost linear but the clipping at 0 is the CORE what makes it non-linear
#       and very powerful
# 
# Two neurons interconnected:
# - Weight 1  and bias 0 on 2nd neuron: Output unchanged
# - Weight 1  and bias 1 on 2nd neuron: Offset vertically (not horizontaly as with 1st neuron)
# - Weight-2  and bias 1 on 2nd neuron: Lower and upper bound for input information on the 2nd neuron


import numpy as np
from print import Print


# Unit Step Function (Heaviside) Activation Function
# - If Input >  0 -> Output = 1                __________
#      Input <= 0 -> Output = 0        ________|
# - Each Inner and Output layer neuron will have this activation function
# - AFTER Input*Weights+Bias calculation
# - Output of this Activation Function will become input for the next neurons
# TODO for Spiking Neural Networks


# Rectified Linear Unit (ReLU) Activation function     /
# - If Input >  0 -> Output = X (Activated)           / Activ.
#      Input <= 0 -> Output = 0 (Deactivated)  ______/
# - Most popular for Hidden layers             Deactiv.
# - Advantages: Linear = Fast, Granular, Solves Vanishing Gradient Problem
# - Disadvantage: Zero Gradient Problem (when gradient is zero at a node, the node is considered dead
#                 since new and old values are the same)
# How does the Activation_ReLU work?
# - Weight tweaks the slope of the linear part
#     - Higher value = Increases slope            '  __/  '
#     - Lower  value = Decreases slope            '  __.  '
#     - Lower  negative value = Increases slope   ' \___  '
#     - Higher negative value = Decreases slope   ' .___  '
# - Bias offsets the start of the linear part horizontally (the point at which the neuron activates/deactivates)
#     - Higher value = Offset to the left
#     - Lower  value = Offset to the right
class Activation_ReLU(Print):
    # Take all inputs and produce activations from all neurons for the entire layer
    def forward(self, inputs, training, verbose=False):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        if verbose:
            self.print_verbose("self.inputs", self.inputs)
            self.print_verbose("self.output", self.output)

    # Perform backpropagation
    # Can be optimized...
    def backward(self, relu, verbose=False):
        # THIS DOES NOT WORK
        # deri_relu = np.zeros_like(relu)
        # deri_relu[relu > 0] = 1.0
        # self.dinputs = relu * deri_relu

        self.dinputs = relu.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        if verbose:
            self.print_verbose("self.dinputs", self.dinputs)
    
    def predictions(self, outputs):
        return outputs



# Softmax:
#                  |--------------'SOFTMAX'----------------|
#       INPUT   ->   EXOPNENTIATE   ->       NORMALIZE       ->   OUTPUT
#         in            exp in           exp in / sum ins
# cat :   |1|           |e**1|           |e1 / (e1+e2+e3)|        |0. 9|
# dog :   |2|   ->      |e**2|      ->   |e2 / (e1+e2+e3)|   ->   |0.24|
# humans: |3|           |e**3|           |e3 / (e1+e2+e3)|        |0.67|
# 
#    Softmax:
#                 Exp(z_ij)
# S_ij =    -------------------
#             \‾ L
#              |    Exp(z_ij)
#             /_ l=1  
# # Overflow Prevention
# - Problem with exponentiation is the explosion of values after exponentiation
#    -> We must trim the values somehow BEFORE EXPONENTIATION becasue we don't need large 
#       numbers as they tell the same information after trimming 
# - One way to do this is to take the maximum value from the batch and subtract it from all the values in the batch
#    -> The effect of this is that the LARGEST value will be 0 -> Exponent of 0 = 1
#       and other values will be negative
#            v = u - max(u)
#    -> Then, all values must fit in between 0 and 1 after Softmax
#           -> Why before 0 and 1?
#              This is benefitial because values will carry the same information and will be much lower, 
#              thus calculations will be faster
class Activation_Softmax(Print):
    def forward(self, inputs, training, verbose=False):
        # Exponentiation & Overflow prevention: Ensure you target the respective batch otherwise data will be modified
        # Specify each batch on the specified axis (axis=0 = columns, axis=1 = rows)
        # Keepdims must be set to True in order to match to the original dimension of the exp_values variable
        # self.inputs = inputs #seems unused
        exponentiated_input_overflow_prevented = np.exp(inputs
                        - np.max(inputs, axis=1, keepdims=True))
        softmax_probabilities_normalized = exponentiated_input_overflow_prevented \
                        / np.sum(exponentiated_input_overflow_prevented, axis=1, keepdims=True)
        self.output = softmax_probabilities_normalized

        if verbose:
            self.print_verbose("exponentiated_input_overflow_prevented", exponentiated_input_overflow_prevented)
            self.print_verbose("softmax_probabilities_normalized", softmax_probabilities_normalized)
            self.print_verbose("self.output[:5]", self.output[:5])
            print(f'PY: NOTE: Initial values should be random for initializing the model')
            print(f'PY: NOTE: Distribution of predicions during model initialization should be 1/3 prediction for everything (values around 0.3)')

    def backward(self, gradients_from_loss_function, verbose=False):

        self.dinputs = np.empty_like(gradients_from_loss_function)

        for i, (softmax_probability_vector, gradient_from_loss_function_vector) in \
            enumerate(zip(self.output, gradients_from_loss_function)):

            # Create a 2D vector of a 1D vector
            softmax_probability_vector = softmax_probability_vector.reshape(-1, 1)

            #                            'left_part'         'right_part'
            # Gradient: dSij/dz_ik = S_ij*Kroneckerdelta_jk - S_ij*S_ik
            # 1) S_ij * Kroneckerdelta_jk
            # left_part_slower = softmax_probability_vector * np.eye(softmax_probability_vector.shape[0])
            # jacobian_left_part_faster = np.diagflat(softmax_probability_vector).copy()

            # 2) S_ij * S_ik
            # jacobian_right_part = np.dot(softmax_probability_vector, softmax_probability_vector.T).copy()

            # Left part - Right part (use the faster version, slower is more unrerstandable, but time-consuming)
            # jacobian_matrix = jacobian_left_part_faster - jacobian_right_part
            jacobian_matrix = np.diagflat(softmax_probability_vector) - \
                np.dot(softmax_probability_vector, softmax_probability_vector.T)

            # Assign the multiplied value to the batch of samples
            self.dinputs[i] = np.dot(jacobian_matrix, gradient_from_loss_function_vector)

            if verbose:
                self.print_verbose("softmax_probability_vector", softmax_probability_vector)
                self.print_verbose("softmax_probability_vector.T", softmax_probability_vector.T)
                # self.print_verbose("jacobian_left_part_faster", jacobian_left_part_faster)
                # self.print_verbose("jacobian_right_part", jacobian_right_part)
                self.print_verbose("jacobian_matrix", jacobian_matrix)
                self.print_verbose("self.dinputs", self.dinputs)

    # Predictions for outputs via searching for an argument with the largest output
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    

# Sigmoid as a Binary Logistic Regression Activation Function
# - f = 1/(1+e^(-x))
#                          ____
#                        /
#                       |
#               _______/
#  - Advanatges:
#  - More reliable because it has more granularity of its output
#       -> this is important to calculate loss (how wrong is the network)
#            -> this can be used in an optimizer, to decrease the loss by tweaking weights and biases
#       -> we can use this to assess how close/far we were to output a 1 or 0
#  - Disadvantage: Vanishing Gradient Problem:
#       -> Occurs with more than one layers with Sigmoid activation
#       -> Since Maximum number of the maximum partial derivative of Sigmoid is 0.25 (not 1 like ReLU), passing 
#          this value through more layers with Sigmoid will cause this value to decrease to zero,
#          thus the partial derivative vanishes and prevents the NN from learning
class Activation_Sigmoid(Print):
    def forward (self, inputs, training, verbose=False):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        if verbose:
            self.print_verbose("self.inputs", self.inputs)
            self.print_verbose("self.output", self.output)

    def backward(self, dvalues, verbose=False):
        self.dinputs = dvalues * (1-self.output) * self.output
        if verbose:
            self.print_verbose("self.dinputs", self.dinputs)

    # Calculate predictions via comparison
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    


# Linear activation function: f(y)=x (For Regression Problems)
# This function exists only for clarity and completeness
class Activation_Linear(Print):
    def forward(self, inputs, verbose=False):
        self.inputs = inputs
        self.output = inputs
        if verbose:
            self.print_verbose("self.inputs", self.inputs)
            self.print_verbose("self.output", self.output)

    def backward(self, dvalues, verbose=False):
        # derivative of f(y)=x is 1 -> 1*dvalues = dvalues
        self.dinputs = dvalues.copy()
        if verbose:
            self.print_verbose("self.dinputs", self.dinputs)

    # Pass outputs as predictions (no filterig here as we are not interested in classifications)
    def predictions(self, outputs):
        return outputs