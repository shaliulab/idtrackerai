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


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, "Input dimension must be 2"
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def Class2Simi(x, mode="cls", mask=None):
    """
    Give a 1d torch tensor with classes in dense format, returns the pairwise similarity matrix liniarized. A mask can
    be applied to discard some elements of the similarity matrix.

    :param x: 1d torch tensor with classes in dense format
    :param mode: 'cls' for classification 'hinge' for clustering
    :param mask: 2d torch tensor with the mask to be applied to the pairwise similarity matrix
    :return: 1d torch tensor with the elements to be considered
    """
    # Convert class label to pairwise similarity
    n = x.nelement()
    assert (n - x.ndimension() + 1) == n, "Dimension of Label is not right"
    expand1 = x.view(-1, 1).expand(n, n)
    expand2 = x.view(1, -1).expand(n, n)
    out = expand1 - expand2
    out[out != 0] = -1  # dissimilar pair: label=-1
    out[out == 0] = 1  # Similar pair: label=1
    if mode == "cls":
        out[out == -1] = 0  # dissimilar pair: label=0
    if mode == "hinge":
        out = out.float()  # hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out
