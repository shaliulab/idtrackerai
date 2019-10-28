import numpy as np
import torch

from idtrackerai.network.utils.metric import Confusion, Timer, AverageMeter
from idtrackerai.network.utils.task import prepare_task_target


def train(epoch, train_loader, learner, network_params):
    """Trains trains a network using a learner, a given train_loader and a set of network_params
    
    :param epoch: current epoch
    :param train_loader: dataloader
    :param learner: learner from learner.py
    :param network_params: networks params from networks_params.py
    :return: losses (tuple) and accuracy
    """

    # Initialize all meters
    data_timer = Timer()
    batch_timer = Timer()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if network_params.loss in ['CEMCL', 'CEMCL_weighted']:
        losses_CE = AverageMeter()
        losses_MCL = AverageMeter()
    confusion = Confusion(network_params.number_of_classes)

    # Setup learner's configuration
    print('\n\n==== Epoch:{0} ===='.format(epoch))
    learner.train()
    learner.step_schedule(epoch)


    # The optimization loop
    data_timer.tic()
    batch_timer.tic()
    if network_params.print_freq>0:  # Enable to print mini-log
        if network_params.loss in ['CEMCL', 'CEMCL_weighted']:
            print('Itr            |Batch time     |Data Time      |Loss         |CE loss      |MCL loss')
        else:
            print('Itr            |Batch time     |Data Time      |Loss')

    for i, (input_, target) in enumerate(train_loader):
        data_time.update(data_timer.toc())  # measure data loading time
        # mask
        mask = None
        if network_params.apply_mask:
            mask = torch.from_numpy(~np.eye(len(target)).astype(bool))
        # Prepare the inputs
        if network_params.use_gpu:
            input_ = input_.cuda()
            target = target.cuda()
            if mask is not None:
                mask = mask.cuda()
        train_target, eval_target = prepare_task_target(target, network_params, mask=mask)

        # Optimization
        if 'weighted' in network_params.loss:
            loss, output = learner.learn(input_, train_target, w_MCL=network_params.w_MCL, mask=mask)
        else:
            loss, output = learner.learn(input_, train_target, mask=mask)


        with torch.no_grad():
            # Update the performance meter
            if network_params.loss in ['CEMCL', 'CEMCL_weighted']:
                confusion.add(output[0], eval_target)
            else:
                confusion.add(output, eval_target)

        # Measure elapsed time
        batch_time.update(batch_timer.toc())
        data_timer.toc()

        # Mini-Logs
        losses.update(loss, input_.size(0))
        if network_params.loss in ['CEMCL', 'CEMCL_weighted']:
            losses_CE.update(output[1], input_.size(0))
            losses_MCL.update(output[2], input_.size(0))
        if network_params.print_freq>0 and ((i%network_params.print_freq==0) or (i==len(train_loader)-1)):
            if 'CEMCL' in network_params.loss:
                print('[{0:6d}/{1:6d}]\t'
                      '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                      '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})'
                      '{loss_CE.val:.3f} ({loss_CE.avg:.3f})\t'
                      '{loss_MCL.val:.3f} ({loss_MCL.avg:.3f})\t'.format(i, len(train_loader),
                                                                           batch_time=batch_time,
                                                                           data_time=data_time,
                                                                           loss=losses,
                                                                           loss_CE=losses_CE, loss_MCL=losses_MCL))
            else:
                print('[{0:6d}/{1:6d}]\t'
                      '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                      '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})'.format(i, len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses))

    # print loss avg
    print(losses.avg)
    # Loss-specific information
    if network_params.loss=='CE':
        print('[Train] ACC: ', confusion.acc())
    elif network_params.loss in ['MCL', 'CEMCL', 'CEMCL_weighted']:
        network_params.cluster2Class = tuple(confusion.optimal_assignment(train_loader.num_classes))  # Save the mapping in network_params to use in eval
        print(network_params.cluster2Class)
        if network_params.out_dim <= 20:  # Avoid to print a large confusion matrix
            confusion.show()
        print('Clustering scores:', confusion.clusterscores())
        print('[Train] ACC: ', confusion.acc())

    if network_params.loss in ['CEMCL', 'CEMCL_weighted']:
        return (losses, losses_CE, losses_MCL), confusion.acc()
    else:
        return (losses, None, None), confusion.acc()
