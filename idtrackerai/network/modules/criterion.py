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


class CE(nn.Module):
    # CrossEntropy (CE) from partial labelling

    def forward(self, prob, one_hot_target):
        return CE_func(prob, one_hot_target)


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    def forward(self, prob1, prob2, simi):
        return MCL_func(prob1, prob2, simi)


class CEMCL(nn.Module):
    # Cross Entropy with Meta Classification (CE-MCL)

    def forward(self, prob, prob1, prob2, one_hot_target, pairwise_target):
        MCL_contrib = MCL_func(prob1, prob2, pairwise_target)
        CE_contrib = CE_func(prob, one_hot_target)
        return MCL_contrib.add(CE_contrib), CE_contrib, MCL_contrib


class CEMCL_weighted(nn.Module):
    # Cross Entropy with Meta Classification (CE-MCL)

    def forward(
        self, prob, prob1, prob2, one_hot_target, pairwise_target, w_MCL=0.5
    ):
        MCL_contrib = MCL_func(prob1, prob2, pairwise_target)
        CE_contrib = CE_func(prob, one_hot_target)
        return (
            MCL_contrib.mul(w_MCL).add(CE_contrib.mul(1 - w_MCL)),
            CE_contrib.mul(1 - w_MCL),
            MCL_contrib.mul(w_MCL),
        )


def MCL_func(prob1, prob2, simi):
    # Meta Classification Likelihood (MCL)
    # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
    assert (
        len(prob1) == len(prob2) == len(simi)
    ), "Wrong input size:{0},{1},{2}".format(
        str(len(prob1)), str(len(prob2)), str(len(simi))
    )
    P = prob1.mul_(prob2)
    P = P.sum(1)
    P.mul_(simi).add_(simi.eq(-1).type_as(P))
    neglogP = -P.add_(1e-7).log_()
    return neglogP.mean()


def CE_func(prob, target):
    # target: one hot vector. All zeros when the class is unknown
    target = target.reshape(prob.size())
    assert len(prob) == len(target), "Wrong input size:{0},{1}".format(
        str(len(prob)), str(len(target))
    )

    prob = prob[target.sum(1).type(torch.bool), :]
    target = target[target.sum(1).type(torch.bool), :]
    neglogP = -target.mul_(prob.add_(1e-7).log_())
    return neglogP.mean()
