import torch

from idtrackerai.network.modules.pairwise import Class2Simi


def prepare_task_target(target, args, mask=None):
    # Prepare the target for different criterion/tasks
    if args.loss == "CE":  # For standard classification
        if "semi" in args.dataset:
            one_hot_targets = target[:, :-1].reshape(-1)
            pairwise_targets = Class2Simi(
                target[:, -1], mode="hinge", mask=mask
            )
            train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
            eval_target = target[:, -1]
        else:
            train_target = eval_target = target
    elif args.loss == "MCL":  # For clustering
        if "semi" in args.dataset:
            one_hot_targets = target[:, :-1].reshape(-1)
            pairwise_targets = Class2Simi(
                target[:, -1], mode="hinge", mask=mask
            )
            train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
            eval_target = target[:, -1]
        else:
            train_target = Class2Simi(target, mode="hinge", mask=mask)
            eval_target = target
    elif args.loss in [
        "CEMCL",
        "CEMCL_weighted",
    ]:  # For semi-supervised clustering
        one_hot_targets = target[:, :-1].reshape(-1)
        pairwise_targets = Class2Simi(target[:, -1], mode="hinge", mask=mask)
        train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
        eval_target = target[:, -1]
    return train_target, eval_target
