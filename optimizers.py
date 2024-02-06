import numpy as np
from print import Print

# Optimizer: Stochastic gradient descent
class Optimizer_SGD(Print):
    def __init__(self, learning_rate=1.0, exponential_decay=0., momentum=0., verbose=False):
        self.learning_rate_initial = learning_rate
        self.learning_rate_current = learning_rate
        self.exponential_decay = exponential_decay
        self.iteration = 0
        self.momentum = momentum
        if verbose:
            self.print_verbose("self.learning_rate_initial", self.learning_rate_initial)
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)
            self.print_verbose("self.exponential_decay", self.exponential_decay)
            self.print_verbose("self.iteration", self.iteration)
            self.print_verbose("self.momentum", self.momentum)

    def update_learning_rate(self, verbose=False):
        if self.iteration:
            self.learning_rate_current = self.learning_rate_initial*(
                    1.0 / (1.0+(self.exponential_decay*self.iteration))
                )
        if verbose:
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)


    # Update weighhts and biases based on the specified learning rate for a specific layer
    def update_weights_biases_basic(self, layer, verbose=False):
        layer.weights += -self.learning_rate_current * layer.dweights
        layer.biases += -self.learning_rate_current * layer.dbiases
        if verbose:
            self.print_verbose("layer.weights", layer.weights)
            self.print_verbose("layer.biases", layer.biases)

    def update_weights_biases(self, layer, verbose=False):
        if self.momentum:
            try:
                layer.momentums_weights = self.momentum*layer.momentums_weights - self.learning_rate_current*layer.dweights
            except AttributeError as e:
                layer.momentums_weights = np.zeros_like(layer.weights)
                layer.momentums_biases = np.zeros_like(layer.biases)
                layer.momentums_weights = self.momentum*layer.momentums_weights - self.learning_rate_current*layer.dweights

            layer.momentums_biases = self.momentum*layer.momentums_biases - self.learning_rate_current*layer.dbiases
            layer.weights += layer.momentums_weights
            layer.biases += layer.momentums_biases
            if verbose:
                self.print_verbose("layer.momentums_weights", layer.momentums_weights)
                self.print_verbose("layer.momentums_biases", layer.momentums_biases)
                self.print_verbose("layer.weights", layer.weights)
                self.print_verbose("layer.biases", layer.biases)
        else:
            self.update_weights_biases_basic(self, layer, verbose=False)

    def increment_epoch_counter(self, verbose=False):
        self.iteration += 1
        if verbose:
            self.print_verbose("self.iteration", self.iteration)


# Adaptive Gradient: - normalized, per-perameter learning rate rather than shared parameter learning
#                    - keep history of previous changes based on which next normalized param updates
#                      (smaller or larger) are made
class Optimizer_AdaGrad(Print):
    def __init__(self, learning_rate=1., exponential_decay=0., epsilon=1e-7, verbose=False):
        self.learning_rate_initial = learning_rate
        self.learning_rate_current = learning_rate
        self.exponential_decay = exponential_decay
        self.iteration = 0
        self.epsilon = epsilon # To prevent division by zero
        if verbose:
            self.print_verbose("self.learning_rate_initial", self.learning_rate_initial)
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)
            self.print_verbose("self.exponential_decay", self.exponential_decay)
            self.print_verbose("self.iteration", self.iteration)
            self.print_verbose("self.epsilon", self.epsilon)

    def update_learning_rate(self, verbose=False):
        if self.iteration:
            self.learning_rate_current = self.learning_rate_initial*(
                    1.0 / (1.0+(self.exponential_decay*self.iteration))
                )
        if verbose:
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)


    # Update weights and biases based on the specified learning rate for a specific layer and its cache
    def update_weights_biases(self, layer, verbose=False):
        try:
            layer.cached_weights += layer.dweights**2
        except AttributeError as e:
            layer.cached_weights = np.zeros_like(layer.weights)
            layer.cached_biases = np.zeros_like(layer.biases)
            layer.cached_weights += layer.dweights**2

        layer.cached_biases += layer.dbiases**2
        layer.weights += - self.learning_rate_current * layer.dweights / (
                            np.sqrt(layer.cached_weights) + self.epsilon)
        layer.biases += - self.learning_rate_current * layer.dbiases / (
                            np.sqrt(layer.cached_biases) + self.epsilon)
        if verbose:
            self.print_verbose("layer.cached_weights", layer.cached_weights)
            self.print_verbose("layer.cached_biases", layer.cached_biases)
            self.print_verbose("layer.weights", layer.weights)
            self.print_verbose("layer.biases", layer.biases)

    def increment_epoch_counter(self, verbose=False):
        self.iteration += 1
        if verbose:
            self.print_verbose("self.iteration", self.iteration)



# Root Mean Square Propagation: - normalized, per-perameter learning rate rather than shared parameter learning
#                               - keep history of previous changes based on which next normalized param updates
#                                 (smaller or larger) are made
#                               - Adds a similar mechanism as momentum
#                               - Uses moving average of the cache
class Optimizer_RMSPropag(Print):
    def __init__(self, learning_rate=0.001, exponential_decay=0., epsilon=1e-7, rho_cache_decay_rate=0.9, verbose=False):
        self.learning_rate_initial = learning_rate
        self.learning_rate_current = learning_rate
        self.exponential_decay = exponential_decay
        self.epsilon = epsilon # To prevent division by zero
        self.rho_cache_decay_rate = rho_cache_decay_rate
        self.iteration = 0
        if verbose:
            self.print_verbose("self.learning_rate_initial", self.learning_rate_initial)
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)
            self.print_verbose("self.exponential_decay", self.exponential_decay)
            self.print_verbose("self.epsilon", self.epsilon)
            self.print_verbose("self.rho_cache_decay_rate", self.rho_cache_decay_rate)
            self.print_verbose("self.iteration", self.iteration)

    def update_learning_rate(self, verbose=False):
        if self.iteration:
            self.learning_rate_current = self.learning_rate_initial*(
                    1.0 / (1.0+(self.exponential_decay*self.iteration))
                )
        if verbose:
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)


    # Update weights and biases based on the specified learning rate for a specific layer and its cache
    def update_weights_biases(self, layer, verbose=False):
        try:
            layer.cached_weights = self.rho_cache_decay_rate * layer.cached_weights + \
                            (1-self.rho_cache_decay_rate) * layer.dweights**2
        except AttributeError as e:
            layer.cached_weights = np.zeros_like(layer.weights)
            layer.cached_biases = np.zeros_like(layer.biases)
            layer.cached_weights = self.rho_cache_decay_rate * layer.cached_weights + \
                            (1-self.rho_cache_decay_rate) * layer.dweights**2

        layer.cached_biases = self.rho_cache_decay_rate * layer.cached_biases + \
                            (1-self.rho_cache_decay_rate) * layer.dbiases**2
        layer.weights += -self.learning_rate_current * \
                        layer.dweights / (np.sqrt(layer.cached_weights) + self.epsilon)
        layer.biases += -self.learning_rate_current * \
                        layer.dbiases / (np.sqrt(layer.cached_biases) + self.epsilon)
        if verbose:
            self.print_verbose("layer.cached_weights", layer.cached_weights)
            self.print_verbose("layer.cached_biases", layer.cached_biases)
            self.print_verbose("layer.weights", layer.weights)
            self.print_verbose("layer.biases", layer.biases)

    def increment_epoch_counter(self, verbose=False):
        self.iteration += 1
        if verbose:
            self.print_verbose("self.iteration", self.iteration)


# Adaptive Momentum: - Optimizer built on RMSProp
#                    - First apply the shared parameter momentum, then per-parameter adaptive learning
#                    - Adds Bias Correction mechanism (not the layer's bias), applied to the cache and momentum
class Optimizer_Adam(Print):
    def __init__(self, learning_rate=0.001, exponential_decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999, verbose=False):
        self.learning_rate_initial = learning_rate
        self.learning_rate_current = learning_rate
        self.exponential_decay = exponential_decay
        self.epsilon = epsilon # To prevent division by zero
        self.rho_cache_decay_rate1 = beta_1
        self.rho_cache_decay_rate2 = beta_2
        self.iteration = 0
        if verbose:
            self.print_verbose("self.learning_rate_initial", self.learning_rate_initial)
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)
            self.print_verbose("self.exponential_decay", self.exponential_decay)
            self.print_verbose("self.epsilon", self.epsilon)
            self.print_verbose("self.rho_cache_decay_rate1", self.rho_cache_decay_rate1)
            self.print_verbose("self.rho_cache_decay_rate2", self.rho_cache_decay_rate2)
            self.print_verbose("self.iteration", self.iteration)

    def update_learning_rate(self, verbose=False):
        if self.iteration:
            self.learning_rate_current = self.learning_rate_initial*(
                    1.0 / (1.0+(self.exponential_decay*self.iteration))
                )
        if verbose:
            self.print_verbose("self.learning_rate_current", self.learning_rate_current)


    # Update weights and biases based on the specified learning rate for a specific layer and its cache
    def update_weights_biases(self, layer, verbose=False):
        try:
            layer.momentums_weights = self.rho_cache_decay_rate1 * layer.momentums_weights \
                            + (1-self.rho_cache_decay_rate1) * layer.dweights
        except AttributeError as e:
            layer.cached_weights = np.zeros_like(layer.weights)
            layer.cached_biases = np.zeros_like(layer.biases)
            layer.momentums_weights = layer.cached_weights.copy()
            layer.momentums_biases = layer.cached_biases.copy()
            layer.momentums_weights = self.rho_cache_decay_rate1*layer.momentums_weights \
                            + (1-self.rho_cache_decay_rate1) * layer.dweights

        layer.momentums_biases = self.rho_cache_decay_rate1*layer.momentums_biases \
                            + (1-self.rho_cache_decay_rate1) * layer.dbiases

        corrected_momentums_weights = layer.momentums_weights \
                            / (1 - self.rho_cache_decay_rate1**(self.iteration+1))
        corrected_momentums_biases = layer.momentums_biases \
                            / (1 - self.rho_cache_decay_rate1**(self.iteration+1))

        layer.cached_weights = self.rho_cache_decay_rate2*layer.cached_weights \
                            + (1-self.rho_cache_decay_rate2) * layer.dweights**2
        layer.cached_biases = self.rho_cache_decay_rate2*layer.cached_biases \
                            + (1-self.rho_cache_decay_rate2) * layer.dbiases**2

        corrected_cached_weights = layer.cached_weights \
                            / (1 - self.rho_cache_decay_rate2**(self.iteration+1))
        corrected_cached_biases = layer.cached_biases \
                            / (1 - self.rho_cache_decay_rate2**(self.iteration+1))

        layer.weights += -self.learning_rate_current*corrected_momentums_weights \
                            / (np.sqrt(corrected_cached_weights) + self.epsilon)
        layer.biases += -self.learning_rate_current*corrected_momentums_biases \
                            / (np.sqrt(corrected_cached_biases) + self.epsilon)
        if verbose:
            self.print_verbose("layer.cached_weights", layer.cached_weights)
            self.print_verbose("layer.cached_biases", layer.cached_biases)
            self.print_verbose("layer.weights", layer.weights)
            self.print_verbose("layer.biases", layer.biases)

    def increment_epoch_counter(self, verbose=False):
        self.iteration += 1
        if verbose:
            self.print_verbose("self.iteration", self.iteration)