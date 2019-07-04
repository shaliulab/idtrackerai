# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import sys
import psutil

import numpy as np
from confapp import conf

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.get_predictions_crossings")

class GetPredictionCrossigns(object):
    def __init__(self, net):
        # Data set
        self.net = net
        self._softmax_probs = []
        self._predictions = []
        self._fc_vectors = []
        self.batch_size = conf.BATCH_SIZE_PREDICTIONS_DCD


    def next_batch_test(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.number_of_image_predicted
        self.number_of_image_predicted += batch_size
        end = self.number_of_image_predicted
        return self.test_images[start:end]

    def predict(self, test_images):
        self.test_images = test_images
        self.number_of_image_predicted = 0
        predictions = []
        while self.number_of_image_predicted < len(test_images):
            predictions.extend(self.net.prediction(self.next_batch_test(batch_size = conf.BATCH_SIZE_PREDICTIONS_DCD)))

        return predictions

    def get_all_predictions(self, test_set):
        # compute maximum number of images given the available RAM
        image_size_bytes = np.prod(test_set.image_size)*4 #XXX:Check it out!!!! np.prod(test_set.image_size**2)*4
        number_of_images_to_be_fitted_in_RAM = len(test_set.test)
        num_images_that_can_fit_in_RAM = 100000#int(psutil.virtual_memory().available*.2/image_size_bytes)
        if number_of_images_to_be_fitted_in_RAM > num_images_that_can_fit_in_RAM:
            logger.debug("There is NOT enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            number_of_predictions_retrieved = 0
            predictions = []
            i = 0
            while number_of_predictions_retrieved < number_of_images_to_be_fitted_in_RAM:
                images = test_set.generate_test_images(interval = (i*num_images_that_can_fit_in_RAM, (i+1)*num_images_that_can_fit_in_RAM))
                predictions.extend(self.predict(images))
                number_of_predictions_retrieved = len(predictions)
                i += 1
        else:
            logger.debug("There is enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            logger.info("getting predictions...")
            test_images = test_set.generate_test_images()
            predictions = self.predict(test_images)
        return predictions

    # def get_predictions_from_images(self, images):
    #     image_size_bytes = np.prod(images.shape[1:])*4
    #     number_of_images_to_be_fitted_in_RAM = len(images)
    #     num_images_that_can_fit_in_RAM = int(psutil.virtual_memory().available*.9/image_size_bytes)
    #     if number_of_images_to_be_fitted_in_RAM > num_images_that_can_fit_in_RAM:
    #         logger.debug("There is NOT enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
    #         number_of_predictions_retrieved = 0
    #         predictions = []
    #         i = 0
    #         while number_of_predictions_retrieved < number_of_images_to_be_fitted_in_RAM:
    #             images_batch = images[i*num_images_that_can_fit_in_RAM : (i+1)*num_images_that_can_fit_in_RAM]
    #             predictions.extend(self.predict(images_batch))
    #             number_of_predictions_retrieved = len(predictions)
    #             i += 1
    #     else:
    #         logger.debug("There is enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
    #         logger.info("getting predictions...")
    #         predictions = self.predict(images)
    #     return predictions
