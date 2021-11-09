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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from confapp import conf
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai.tracker.dataset.identification_dataloader import (
    get_training_data_loaders,
)
from idtrackerai.tracker.dataset.identification_dataset import (
    split_data_train_and_validation,
)
from idtrackerai.tracker.network.stop_training_criteria import (
    Stop_Training,
)
from idtrackerai.tracker.network.trainer import (
    TrainIdentification,
)
from idtrackerai.network.utils.utils import fc_weights_reinit


logger = logging.getLogger("__main__.pre_trainer")


def pre_train_global_fragment(
    video,
    identification_model,
    learner_class,
    network_params,
    pretraining_global_fragment,
    list_of_fragments,
    global_epoch,
):
    """Performs pretraining on a single global fragments

    Parameters
    ----------
    net : <ConvNetwork obejct>
        an instance of the class :class:`~idCNN.ConvNetwork`
    pretraining_global_fragment : <GlobalFragment object>
        an instance of the class :class:`~globalfragment.GlobalFragment`
    list_of_fragments : <ListOfFragments object>
        an instance of the class :class:`~list_of_fragments.ListOfFragments`
    global_epoch : int
        global counter of the training epoch in pretraining
    check_for_loss_plateau : bool
        if True the stopping criteria (see :mod:`~stop_training_criteria`) will
        automatically stop the training in case the loss functin computed for
        the validation set of images reaches a plateau
    store_accuracy_and_error : bool
        if True the values of the loss function, accuracy and individual
        accuracy will be stored
    save_summaries : bool
        if True tensorflow summaries will be generated and stored to allow
        tensorboard visualisation of both loss and activity histograms
    store_training_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`
    store_validation_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`
    print_flag : bool
        if True additional information are printed in the terminal
    plot_flag : bool
        if True training and validation loss, accuracy and individual accuracy
        are plot in a graph at the end of the training session
    batch_size : int
        size of the batch of images used for training
    canvas_from_GUI : matplotlib figure canvas
        canvas of the matplotlib figure initialised in
        :class:`~tracker.Tracker` used to update the figure in the GUI
        visualisation of pretraining

    Returns
    -------
    <ConvNetwork object>
        network with updated parameters after training
    float
        ration of images used for pretraining over the total number of
        available images
    int
        global epoch counter updated after the training session
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the training set of labelled
        images
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the validation set of labelled
        images
    <ListOfFragments objects>
        list of instances of the class :class:`~fragment.Fragment`
    """
    # Get images and labels from the current global fragment
    images, labels = pretraining_global_fragment.get_images_and_labels(
        list_of_fragments.identification_images_file_paths, scope="pretraining"
    )

    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    logger.debug("images: {} {}".format(images.shape, images.dtype))
    logger.debug("labels: %s" % str(labels.shape))

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(
        video, train_data, val_data
    )

    # Set criterion
    logger.info("Setting training criterion")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    # Re-initialize fully-connected layers
    identification_model.apply(fc_weights_reinit)

    # Send model and criterion to GPU
    if network_params.use_gpu:
        logger.info("Sending model and criterion to GPU")
        torch.cuda.set_device(0)
        cudnn.benchmark = True  # make it train faster
        identification_model = identification_model.cuda()
        criterion = criterion.cuda()

    # Set optimizer
    logger.info("Setting optimizer")
    optimizer = torch.optim.__dict__[network_params.optimizer](
        identification_model.parameters(), **network_params.optim_args
    )

    # Set scheduler
    logger.info("Setting scheduler")
    scheduler = MultiStepLR(
        optimizer, milestones=network_params.schedule, gamma=0.1
    )

    # Set learner
    logger.info("Setting the learner")
    learner = learner_class(
        identification_model, criterion, optimizer, scheduler
    )

    # Set stopping criteria
    logger.info("Setting the stopping criteria")
    # set criteria to stop the training
    stop_training = Stop_Training(
        network_params.number_of_classes,
        check_for_loss_plateau=True,
        first_accumulation_flag=video is None or video.accumulation_step == 0,
    )

    logger.info("Training identification network")
    trainer = TrainIdentification(
        learner, train_loader, val_loader, network_params, stop_training
    )

    logger.info("Identification network trained")

    pretraining_global_fragment.update_individual_fragments_attribute(
        "_used_for_pretraining", True
    )
    global_epoch += stop_training.epochs_completed

    ratio_of_pretrained_images = (
        list_of_fragments.compute_ratio_of_images_used_for_pretraining()
    )
    logger.debug(
        "ratio of images used during pretraining: "
        "%.4f (if higher than %.2f we stop pretraining)"
        % (ratio_of_pretrained_images, conf.MAX_RATIO_OF_PRETRAINED_IMAGES)
    )

    return (
        identification_model,
        ratio_of_pretrained_images,
        global_epoch,
        list_of_fragments,
        trainer.best_model_path,
    )
