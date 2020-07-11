import numpy as np
import torch

from idtrackerai.network.utils.metric import Confusion, AverageMeter
from idtrackerai.network.utils.task import prepare_task_target


def evaluate(eval_loader, model, label, args, learner=None):

    with torch.no_grad():
        # Initialize all meters
        losses = AverageMeter()
        if args.loss in ["CEMCL", "CEMCL_weighted"]:
            losses_CE = AverageMeter()
            losses_MCL = AverageMeter()
        confusion = Confusion(args.number_of_classes)

    print("---- Evaluation ----")
    if learner is not None:
        learner.eval()
    if model is not None:
        model.eval()
    for i, (input_, target) in enumerate(eval_loader):

        # mask
        mask = None
        if args.apply_mask:
            mask = torch.from_numpy(~np.eye(len(target)).astype(bool))
        # Prepare the inputs
        if args.use_gpu:
            with torch.no_grad():
                input_ = input_.cuda()
                target = target.cuda()
                if mask is not None:
                    mask = mask.cuda()
        train_target, eval_target = prepare_task_target(
            target, args, mask=mask
        )

        with torch.no_grad():
            if learner is not None:
                # Optimization
                if "weighted" in args.loss:
                    loss, output = learner.forward_with_criterion(
                        input_, train_target, w_MCL=args.w_MCL, mask=mask
                    )
                else:
                    loss, output = learner.forward_with_criterion(
                        input_, train_target, mask=mask
                    )

                losses.update(loss, input_.size(0))
                if args.loss in ["CEMCL", "CEMCL_weighted"]:
                    losses_CE.update(output[1], input_.size(0))
                    losses_MCL.update(output[2], input_.size(0))

        # Inference
        if model is not None:
            output = model(input_)

        # print(output.shape, eval_target.shape)

        # Update the performance meter
        with torch.no_grad():
            confusion.add(output, eval_target)

    # print loss avg
    print(losses.avg)
    # Loss-specific information
    KPI = 0
    if args.loss == "CE":
        KPI = confusion.acc()
        print("[{}] ACC: ".format(label), KPI)
    elif args.loss in ["MCL", "CEMCL", "CEMCL_weighted"]:
        confusion.optimal_assignment(
            eval_loader.num_classes, args.cluster2Class
        )
        if args.out_dim <= 20:
            confusion.show()
        print("Clustering scores:", confusion.clusterscores())
        KPI = confusion.acc()
        print("[{}] ACC: ".format(label), KPI)

    if learner is not None:
        if args.loss in ["CEMCL", "CEMCL_weighted"]:
            return (losses, losses_CE, losses_MCL), confusion.acc()
        else:
            return (losses, None, None), confusion.acc()
    else:
        return (None, None, None), confusion.acc()
