import numpy as np
from print import Print


# Common Accuracy class - Accuracies are calculated separately, this class unifies workflow
class Accuracy:
    # We always calculate accuracy as a mean value of comparisons
    def calculate(self, predictions, y):
        comparisons = self.compare_against(predictions, y)
        accuracy = np.mean(comparisons)

        # Accumulated sum of comparisons over all batches to be used with calculate_accumulated
        self.all_comparisons_batches_accumulated += np.sum(comparisons)
        self.all_comparisons_count += len(comparisons)

        return accuracy
    
    # Accumulated loss for accumulated batches of inputs
    def calculate_accumulated(self):
        # Calculate mean accuracy
        accuracy = self.all_comparisons_batches_accumulated / self.all_comparisons_count
        return accuracy

    # Reset accumulated losses
    def reset_calculate_accumulated(self):
        self.all_comparisons_batches_accumulated = 0
        self.all_comparisons_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.accuracy_precision = None

    # Calculate precision after constructor has been executed
    def set_precision(self, y, reinit=False):
        # Recalculate precision whenever needed when reinit = True or during initialization
        if self.accuracy_precision is None or reinit:
            self.accuracy_precision = np.std(y) / 250

    # Compare predictions versus the ground truth values and precision
    def compare_against(self, predictions, y):
        return np.absolute(predictions-y) < self.accuracy_precision

class Accuracy_Categorical(Accuracy):
    def set_precision(self, y):
        pass

    # Compare predictions versus the ground truth values
    def compare_against(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1) # Column
        return predictions == y