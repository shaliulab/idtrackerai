from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from idtrackerai.constants import  MAX_FLOAT, LEARNING_PERCENTAGE_DIFFERENCE_2_DCD, \
                    LEARNING_PERCENTAGE_DIFFERENCE_1_DCD, OVERFITTING_COUNTER_THRESHOLD_DCD, \
                    MAXIMUM_NUMBER_OF_EPOCHS_DCD
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.stop_training_criteria_crossings")

class Stop_Training(object):
    """Stops the training of the network according to the conditions specified
    in __call__
    """
    def __init__(self, epochs_before_checking_stopping_conditions = 10, check_for_loss_plateau = True):
        self.num_epochs = MAXIMUM_NUMBER_OF_EPOCHS_DCD #maximal num of epochs
        self.number_of_classes = 2
        self.epochs_before_checking_stopping_conditions = epochs_before_checking_stopping_conditions
        self.overfitting_counter = 0 #number of epochs in which the network is overfitting before stopping the training
        self.check_for_loss_plateau = check_for_loss_plateau #bool: if true the training is stopped if the loss is not decreasing enough

    def __call__(self, loss_accuracy_training, loss_accuracy_validation, epochs_completed):
        #check that the model did not diverged (nan loss).
        if epochs_completed > 0 and (np.isnan(loss_accuracy_training.loss[-1]) or np.isnan(loss_accuracy_validation.loss[-1])):
            logger.warn('The model diverged with loss NaN, falling back to detecting crossings with the model area')
            return True
        #check if it did not reached the epochs limit
        if epochs_completed > self.num_epochs-1:
            logger.warn('The number of epochs completed is larger than the number of epochs set for training, we stop the training')
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
                if self.overfitting_counter >= OVERFITTING_COUNTER_THRESHOLD_DCD and not self.first_accumulation_flag:
                    logger.info('Overfitting')
                    return True
            else:
                self.overfitting_counter = 0
            #check if the error is not decreasing much
            if self.check_for_loss_plateau:
                if np.abs(losses_difference) < LEARNING_PERCENTAGE_DIFFERENCE_2_DCD * 10**(int(np.log10(current_loss))-1):
                    logger.info('The losses difference is very small, we stop the training\n')
                    return True
            # if the individual accuracies in validation are 1. for all the animals
            if list(loss_accuracy_validation.individual_accuracy[-1]) == list(np.ones(self.number_of_classes)):
                logger.info('The individual accuracies in validation is 1. for all the classes, we stop the training\n')
                return True
            # if the validation loss is 0.
            if previous_loss == 0. or current_loss == 0.:
                logger.info('The validation loss is 0., we stop the training')
                return True

        return False
