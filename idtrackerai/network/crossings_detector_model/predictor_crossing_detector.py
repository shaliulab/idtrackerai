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

import torch

from idtrackerai.network.data_loaders.crossings_dataloader import get_test_data_loader

import logging
logger = logging.getLogger("__main__.get_predictions_crossings")


class GetPredictionCrossigns(object):
    def __init__(self, video, model, blobs, network_params):
        # Data set
        self.model = model
        self.network_params = network_params
        self.loader = get_test_data_loader(video, blobs)
        self._predictions = []

    def get_all_predictions(self):
        self.model.eval()
        predictions = []
        for i, (input_, target) in enumerate(self.loader):
            # Prepare the inputs
            if self.network_params.use_gpu:
                with torch.no_grad():
                    input_ = input_.cuda()

            # Inference
            output = self.model(input_)
            pred = output.argmax(1)  # find the predicted class

            predictions.extend(pred.cpu().numpy())

        del self.loader
        return predictions


    # def next_batch_test(self, batch_size):
    #     """Return the next `batch_size` examples from this data set."""
    #     start = self.number_of_image_predicted
    #     self.number_of_image_predicted += batch_size
    #     end = self.number_of_image_predicted
    #     return self.test_images[start:end]
    #
    # def predict(self, test_images):
    #     self.test_images = test_images
    #     self.number_of_image_predicted = 0
    #     predictions = []
    #     while self.number_of_image_predicted < len(test_images):
    #         predictions.extend(self.net.prediction(self.next_batch_test(batch_size = conf.BATCH_SIZE_PREDICTIONS_DCD)))
    #
    #     return predictions

    # def get_all_predictions(self, test_set):
    #     predictions = self.predict(test_set.images)
    #     return predictions


