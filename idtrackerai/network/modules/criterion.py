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
