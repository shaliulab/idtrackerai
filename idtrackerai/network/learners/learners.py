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
        model = models.__dict__[learner_params.architecture](out_dim=learner_params.number_of_classes,
                                                             input_shape=learner_params.image_size)
        return model

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets, **kwargs):
        out = self.forward(inputs)
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
        self.scheduler.step(self.epoch)
        for param_group in self.optimizer.param_groups:
            print('LR:',param_group['lr'])

    def save_model(self, savename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', savename)
        self.model_path = savename + '.pth'
        torch.save(model_state, self.model_path)
        print('=> Done')


    def snapshot(self, savename, KPI=-1):
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        checkpoint = {
            'epoch': self.epoch,
            'model': model_state,
            'optimizer': optim_state
        }
        print('=> Saving checkpoint to:', savename+'.checkpoint.pth')
        torch.save(checkpoint, savename+'.checkpoint.pth')
        print('=> Done')
        if KPI >= self.KPI:
            print('=> New KPI:', KPI, 'previous KPI:', self.KPI)
            self.KPI = KPI
            self.save_model(savename + '.model')
        return self.model_path

    def resume(self, resumefile):
        print('=> Loading checkpoint:', resumefile)
        checkpoint = torch.load(resumefile, map_location=lambda storage, loc: storage)  # Load to CPU as the default!
        self.epoch = checkpoint['epoch']
        print('=> resume epoch:', self.epoch)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> Done')
        return self.epoch
