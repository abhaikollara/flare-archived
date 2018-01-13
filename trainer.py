import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm



def _to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return list(x)

def _get_optimizer(optimizer, model):
    aliases = {
        'adadelta'  : optim.Adadelta,
        'adagrad'   : optim.Adagrad,
        'adam'      : optim.Adam,
        'adamax'    : optim.Adamax,
        'asgd'      : optim.ASGD,
        'lbfgs'     : optim.LBFGS,
        'rms'       : optim.RMSprop,
        'rprop'     : optim.Rprop,
        'sgd'       : optim.SGD,
    }

    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    else:
        if optimizer in aliases.keys():
            return aliases[optimizer](model.parameters())
        else:
            raise TypeError('Optimizer not understood')


class dataset(Dataset):

    def __init__(self, inputs, targets):
        super(dataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

        if len(set([len(x) for x in self.inputs])) != 1:
            raise ValueError('Inputs must have equal n_samples dimension')

        if len(inputs[0]) != len(targets):
            raise ValueError('Inputs and targets must have equal n_samples dimension')

    def __len__(self):
            return self.inputs[0].size()[0]

    def __getitem__(self, idx):
        return [x[idx] for x in self.inputs], self.targets[idx]


class Trainer(object):

    def __init__(self, model, loss, optimizer):
        self.model = model
        self.optimizer = _get_optimizer(optimizer, model)
        self.loss_func = loss

    def train(self, inputs, targets, batch_size=1, epochs=1,
              validation_split=0.0, validation_data=None, shuffle=True):
        '''
                TODO : 
                2. Shuffle
                3. Class weight
        '''

        inputs = _to_list(inputs)

        if validation_split > 0.0:
            split_size = int(len(inputs) * validation_split)
            train_dataset = dataset([x[-split_size:] for x in inputs], targets[:-split_size])
            validation_data = ([x[:-split_size] for x in inputs], targets[:-split_size])
        else:
            train_dataset = dataset(inputs, targets)

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)

        loss = Variable(torch.Tensor([0.]))

        for epoch in range(epochs):
            print("Epoch", str(epoch), "of", str(epochs))
            for batch in tqdm(train_data_loader, postfix={'loss':loss.data.cpu().numpy()[0]}):
                batch_inputs, batch_targets = batch
                loss = self.train_batch(batch_inputs, batch_targets)

            if validation_data:
                self.evaluate(validation_data[0], validation_data[1])


    def evaluate(self, inputs, targets, batch_size=1, metrics=['accuracy']):
        #TODO : 1. Metrics

        inputs = _to_list(inputs)
        valid_dataset = dataset(inputs, targets)
        valid_data_loader = DataLoader(
                    valid_dataset, batch_size=batch_size, shuffle=False)
        losses = []
        for batch in tqdm(valid_data_loader):
            batch_inputs, batch_targets = batch
            batch_loss = self.evaluate_batch(batch_inputs, batch_targets)
            losses.append(batch_loss)

        mean_loss = torch.mean(torch.cat(losses))
        print('Loss: ', mean_loss.data.cpu().numpy()[0])

    def predict(self, inputs, batch_size=1):
        pass

    def train_batch(self, inputs, targets, class_weight=None):

        self.optimizer.zero_grad()
        input_batch = [Variable(x) for x in inputs]
        target_batch = Variable(targets)
        self.model.train()

        # TODO : Make inputs and targets accept np nd-arrayss
        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        loss = self.loss_func(y, target_batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate_batch(self, inputs, targets, metrics=['accuracy']):
        input_batch = [Variable(x, volatile=True) for x in inputs]
        target_batch = Variable(targets, volatile=True)
        self.model.eval()
        # TODO : Make inputs and targets accept np nd-arrayss

        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        loss = self.loss_func(y, target_batch)
        return loss

    def predict_batch(self, inputs):
        pass
