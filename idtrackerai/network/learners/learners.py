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

import torch
import torch.nn as nn

import idtrackerai.network.models.pytorch_architectures as models

# This file provides the template Learner. The Learner is used in training/evaluation loop
# The Learner implements the training procedure for specific task.
# The default Learner is from classification task.


class Learner_Classification(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(Learner_Classification, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.KPI = -1  # An non-negative index, larger is better.
        self.model_path = None

    @staticmethod
    def create_model(learner_params):
        # This function create the model for specific learner
        # The create_model(), forward_with_criterion(), and learn() are task-dependent
        # Do surgery to generic model if necessary
        model = models.__dict__[learner_params.architecture](
            out_dim=learner_params.number_of_classes,
            input_shape=learner_params.image_size,
        )
        return model

    @staticmethod
    def load_model(learner_params, scope=""):
        model = Learner_Classification.create_model(learner_params)
        if scope == "knowledge_transfer":
            model_path = learner_params.knowledge_transfer_model_file
        else:
            model_path = learner_params.load_model_path

        print(
            "=> Load model weights:", model_path
        )  # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
        # model_state = torch.load(
        #     model_path, map_location=lambda storage, loc: storage
        # )  # Load to CPU as the default!
        model_state = torch.load(model_path)
        model.load_state_dict(
            model_state, strict=True
        )  # The pretrained state dict doesn't need to fit the model
        print("=> Load Done")
        return model

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets, **kwargs):
        out = self.forward(inputs)
        targets = targets.long()
        return self.criterion(out, targets), out

    def learn(self, inputs, targets, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            loss, out = self.forward_with_criterion(inputs, targets, **kwargs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, out

    def step_schedule(self, epoch):
        self.epoch = epoch
        self.scheduler.step()
        # for param_group in self.optimizer.param_groups:
        # print("LR:", param_group["lr"])

    def save_model(self, savename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        # print("=> Saving model to:", savename)
        self.model_path = savename + ".pth"
        torch.save(model_state, self.model_path)
        # print("=> Done")

    def snapshot(self, savename, KPI=-1):
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        checkpoint = {
            "epoch": self.epoch,
            "model": model_state,
            "optimizer": optim_state,
        }
        # print("=> Saving checkpoint to:", savename + ".checkpoint.pth")
        torch.save(checkpoint, savename + ".checkpoint.pth")
        self.save_model(savename + ".model")
        return self.model_path

    def resume(self, resume_file):
        print("=> Loading checkpoint:", resume_file)
        # checkpoint = torch.load(
        #     resume_file, map_location=lambda storage, loc: storage
        # )  # Load to CPU as the default!
        checkpoint = torch.load(resume_file)
        self.epoch = checkpoint["epoch"]
        print("=> resume epoch:", self.epoch)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> Done")
        return self.epoch
