from __future__ import absolute_import, division, print_function
import sys
import numpy as np

MAX_FLOAT = sys.float_info[0]
LEARNING_PERCENTAGE_DIFFERENCE = .01
OVERFITTING_COUNTER_THRESHOLD = 10
MAXIMUM_NUMBER_OF_EPOCHS = 10000

class Stop_Training():
    """Stops the training of the network according to the conditions specified
    in __call__
    """
    def __init__(self, number_of_animals, epochs_before_checking_stopping_conditions = 10, check_for_loss_plateau = True, first_accumulation_flag = False):
        self.num_epochs = MAXIMUM_NUMBER_OF_EPOCHS #maximal num of epochs
        self.number_of_animals = number_of_animals
        self.epochs_before_checking_stopping_conditions = epochs_before_checking_stopping_conditions
        self.overfitting_counter = 0 #number of epochs in which the network is overfitting before stopping the training
        self.check_for_loss_plateau = check_for_loss_plateau #bool: if true the training is stopped if the loss is not decreasing enough
        self.first_accumulation_flag = first_accumulation_flag

    def __call__(self, loss_accuracy_training, loss_accuracy_validation, epochs_completed):
        #check that the model did not diverged (nan loss).

        if epochs_completed > 0 and np.isnan(loss_accuracy_training.loss[-1]):
            raise ValueError("The model diverged.")
            return True
        #check if it did not reached the epochs limit
        if epochs_completed > self.num_epochs-1:
            print('The number of epochs completed is larger than the number of epochs set for training, we stop the training\n')
            return True
        #check that the model is not overfitting or if it reached a stable saddle (minimum)
        if epochs_completed > self.epochs_before_checking_stopping_conditions:
            current_loss = loss_accuracy_validation.loss[-1]
            previous_loss = np.nanmean(loss_accuracy_validation.loss[-self.epochs_before_checking_stopping_conditions:-1])
            #The validation loss in the first 10 epochs could have exploded but being decreasing.
            if np.isnan(previous_loss): previous_loss = MAX_FLOAT
            losses_difference = (previous_loss-current_loss)
            #check overfitting
            if losses_difference < 0.:
                self.overfitting_counter += 1
                if self.overfitting_counter >= OVERFITTING_COUNTER_THRESHOLD and not self.first_accumulation_flag:
                    print('Overfitting\n')
                    return True
            else:
                self.overfitting_counter = 0
            #check if the error is not decreasing much
            if self.check_for_loss_plateau:
                if np.abs(losses_difference) < LEARNING_PERCENTAGE_DIFFERENCE*10**(int(np.log10(current_loss))-1):
                    print('The losses difference is very small, we stop the training\n')
                    return True
            # if the individual accuracies in validation are 1. for all the animals
            if list(loss_accuracy_validation.individual_accuracy[-1]) == list(np.ones(self.number_of_animals)):
                print('The individual accuracies in validation is 1. for all the individuals, we stop the training\n')
                return True
            # if the validation loss is 0.
            if previous_loss == 0. or current_loss == 0.:
                print('The validation loss is 0., we stop the training')
                return True

        return False
