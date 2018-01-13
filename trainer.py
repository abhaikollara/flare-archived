import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


def _to_list(x):
    '''
        Used to ensure that model input is
        always a list, even if single input is required
    '''
    if isinstance(x, (list, tuple)):
        return x
    else:
        return list(x)


def _get_optimizer(optimizer, model):
    aliases = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rms': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    else:
        if optimizer in aliases.keys():
            return aliases[optimizer](filter(lambda p: p.requires_grad,
                                            model.parameters()))
        else:
            raise TypeError('Optimizer not understood')


class dataset(Dataset):

    def __init__(self, inputs, targets=None):
        super(dataset, self).__init__()
        self.inputs = inputs
        self.targets = targets
        if len(set([len(x) for x in self.inputs])) != 1:
            raise ValueError('Inputs must have equal n_samples dimension')

        if targets is not None:
            if len(self.inputs[0]) != len(self.targets):
                raise ValueError(
                    'Inputs and targets must have equal n_samples dimension')

    def __len__(self):
        return self.inputs[0].size()[0]

    def __getitem__(self, idx):
        if self.targets is not None:
            return [x[idx] for x in self.inputs], self.targets[idx]
        else:
            return [x[idx] for x in self.inputs]


class Trainer(object):

    def __init__(self, model, loss, optimizer):
        """ A trainer utility for PyTorch modules

        # Arguments:
            model: An instance of torch.nn.Module
            loss: A PyTorch loss function
            optimizer: String (name of optimizer) or optimizer object

        # Raises:
            TypeError: If optimizer is invalid
        """
        self.model = model
        self.optimizer = _get_optimizer(optimizer, model)
        self.loss_func = loss

    def train(self, inputs, targets, batch_size=1, epochs=1,
              validation_split=0.0, validation_data=None, shuffle=True):
        """Trains the model for a fixed number of epochs

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            targets: torch.Tensor. Target values/classes
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model.
            validation_split: float (0. < x < 1.)
                Fraction of data to use as validation data
            validation_data: tuple(input_valid, target_valid)
            shuffle: boolean. Whether to shuffle data at each epoch


        #Raises
            ValueError: If the number of samples in inputs and
                        targets are not equal
        """

        inputs = _to_list(inputs)

        if validation_split > 0.0:
            split_size = int(len(inputs[0]) * validation_split)
            train_dataset = dataset([x[:-split_size]
                                     for x in inputs], targets[:-split_size])
            validation_data = ([x[-split_size:]
                                for x in inputs], targets[-split_size:])
        else:
            train_dataset = dataset(inputs, targets)

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)

        loss = Variable(torch.Tensor([0.]))  # Required for progress bar

        for epoch in range(epochs):
            print("Epoch", str(epoch), "of", str(epochs))
            for batch in tqdm(train_data_loader,
                              postfix={'loss': loss.data.cpu().numpy()[0]}):
                batch_inputs, batch_targets = batch
                loss = self.train_batch(batch_inputs, batch_targets)

            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                print("Validation loss ", val_loss.data.cpu().numpy()[0])

    def evaluate(self, inputs, targets, batch_size=1, metrics=['accuracy']):
        """Computes and prints the loss on data
            batch by batch without optimizing

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            targets: torch.Tensor. Target values/classes
            batch_size: int. Number of samples per gradient update.
            metrics : ## TODO ##

        #Raises
            ValueError: If the number of samples in inputs and
                        targets are not equal
        """

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
        return mean_loss

    def predict(self, inputs, batch_size=1, classes=False):
        """Generates output predictions batch
           by batch for the input samples.

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            batch_size: integer. Number of samples per batch
            classes: boolean. Whether to return class predictions

        # Returns
            A torch variable of predictions

        #Raises
            ValueError: If the number of samples in inputs are
                        not equal
        """
        inputs = _to_list(inputs)
        predict_dataset = dataset(inputs)
        predict_data_loader = DataLoader(
            predict_dataset, batch_size=batch_size, shuffle=False)

        preds = []
        for batch in tqdm(predict_data_loader):
            pred = self.predict_batch(batch, classes=classes)
            preds.append(pred)

        return torch.cat(preds, dim=-1)

    def train_batch(self, inputs, targets):
        """ Single gradient update over one batch of samples

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            targets: torch.Tensor. Target values/classes

        # Returns
            Scalar training loss as torch Variable
        """
        inputs = _to_list(inputs)

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
        """Evaluates the model over a single batch of samples.

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            targets: torch.Tensor. Target values/classes

        # Returns
            Scalar test loss as torch variable

        """
        inputs = _to_list(inputs)

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

    def predict_batch(self, inputs, classes=False):
        """Returns predictions for a single batch of samples.

        # Arguments
            inputs: torch.Tensor or list of torch.Tensor
            classes: boolean. Whether to return class predictions
        # Returns
            A torch variable of predictions
        """
        inputs = _to_list(inputs)
        input_batch = [Variable(x, volatile=True) for x in inputs]

        # TODO : Make inputs and targets accept np nd-arrayss

        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        if classes:
            return torch.max(y, -1)[1]
        else:
            return y
