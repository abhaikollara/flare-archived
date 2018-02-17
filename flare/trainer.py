import inspect

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
        return [x]

def _wrap_in_tensor(x):
    if torch.is_tensor(x):
        return x
    if issubclass(x.dtype.type, np.floating):
        return torch.FloatTensor(x)
    elif issubclass(x.dtype.type, np.integer):
        return torch.LongTensor(x)
    else:
        raise TypeError('Input array must be valid numpy arrays')

class dataset(Dataset):

    def __init__(self, inputs, targets=None):
        super(dataset, self).__init__()
        self.inputs = inputs
        self.targets = targets
        if len(set(len(x) for x in self.inputs)) != 1:
            raise ValueError('Inputs must have equal n_samples dimension.')

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
            optimizer: An instance of torch.optim object
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss

    def train(self, inputs, targets, batch_size=1, epochs=1,
              validation_split=0.0, validation_data=None, shuffle=True, disable_progbar=False):
        """Trains the model for a fixed number of epochs

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            targets: Target values/classes. A numpy array or a torch tensor.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model.
            validation_split: float (0. < x < 1.)
                Fraction of data to use as validation data. This
                takes precedence of validation_data
            validation_data: tuple(input_data, target_data)
            shuffle: boolean. Whether to shuffle data at each epoch

        #Raises
            ValueError: If the number of samples in inputs and
                        targets are not equal
        """

        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]
        targets = _wrap_in_tensor(targets)

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

        self.train_on_generator(
            train_data_loader, epochs=epochs, validation_data=validation_data)

    def train_on_generator(self, generator, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None):
        """Trains the model on data generator

        # Arguments
            generator : A user created data generator or
                torch.utils.data.DataLoader. The generator should
                yield and iterable of (input_data, target_data)
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from generator before declaring one epoch
                finished and starting the next epoch. Must be specified
                for user created generator
            epochs: int. Number of epochs to train the model.
            validation_data: It can be any of the following
                tuple(input_data, target_data),
                generator yielding a tuple(input_data, target_data),
                torch.utils.DataLoader,
            validation_steps: Total number of steps to yield from
                validation generator. Required only if 
                you are using a generator for validation data.

        #Raises
            ValueError: If the number of samples in inputs and
                        targets are not equal
            
            AssertionError: If generator is a user defined generator and 
                steps_per_epoch is not specified
        """
        if isinstance(generator, DataLoader):
            for epoch in range(epochs):
                print('Epoch', str(epoch), 'of', str(epochs))
                for batch in tqdm(generator):
                    batch_inputs, batch_targets = batch
                    _ = self.train_batch(batch_inputs, batch_targets)

                if validation_data is not None:
                    if isinstance(validation_data, DataLoader) or inspect.isgenerator(validation_data):
                        self.evaluate_on_generator(validation_data, steps_per_epoch=validation_steps)
                    else:
                        self.evaluate(validation_data[0], validation_data[1])
        else:
            assert steps_per_epoch is not None
            for epoch in range(epochs):
                print('Epoch', str(epoch), 'of', str(epochs))
                for _ in tqdm(range(steps_per_epoch)):
                    batch_inputs, batch_targets = next(generator)
                    _ = self.train_batch(batch_inputs, batch_targets)

                if validation_data is not None:
                    if isinstance(validation_data, DataLoader) or inspect.isgenerator(validation_data):
                        self.evaluate_on_generator(
                            validation_data, validation_steps)
                    else:
                        self.evaluate(validation_data[0], validation_data[1])

    def train_batch(self, inputs, targets):
        """ Single gradient update over one batch of samples

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            targets: Target values/classes. A numpy array or a torch tensor.

        # Returns
            Scalar training loss as torch tensor
        """
        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]
        targets = _wrap_in_tensor(targets)

        self.optimizer.zero_grad()
        input_batch = [Variable(x) for x in inputs]
        target_batch = Variable(targets)
        self.model.train()

        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        loss = self.loss_func(y, target_batch)
        loss.backward()
        self.optimizer.step()
        return loss.data

    def evaluate(self, inputs, targets, batch_size=1):
        """Computes and prints the loss on data
            batch by batch without optimizing

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            targets: Target values/classes. A numpy array or a torch tensor.
            batch_size: int. Number of samples per gradient update.

        #Raises
            ValueError: If the number of samples in inputs and
                        targets are not equal
        """

        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]
        targets = _wrap_in_tensor(targets)

        valid_dataset = dataset(inputs, targets)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=batch_size)

        self.evaluate_on_generator(valid_data_loader)

    def evaluate_on_generator(self, generator, steps_per_epoch=None):
        """Evaluates the model on data generator

        # Arguments
            generator : A user created data generator or
                torch.utils.data.DataLoader. The generator should
                yield and iterable of (input_data, target_data)
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from generator before declaring one epoch
                finished and starting the next epoch. Must be specified
                for user created generator

        #Raises
            AssertionError: If generator is a user defined generator and 
                steps_per_epoch is not specified
        """
        if isinstance(generator, DataLoader):
            for batch in tqdm(generator):
                batch_inputs, batch_targets = batch
                _ = self.evaluate_batch(batch_inputs, batch_targets)
        else:
            assert steps_per_epoch is not None
            for _ in tqdm(range(steps_per_epoch)):
                batch_inputs, batch_targets = next(generator)
                _ = self.evaluate_batch(batch_inputs, batch_targets)

    def evaluate_batch(self, inputs, targets):
        """Evaluates the model over a single batch of samples.

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            targets: Target values/classes. A numpy array or a torch tensor.

        # Returns
            Scalar test loss as torch tensor

        """
        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]
        targets = _wrap_in_tensor(targets)

        input_batch = [Variable(x, volatile=True) for x in inputs]
        target_batch = Variable(targets, volatile=True)
        self.model.eval()

        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        loss = self.loss_func(y, target_batch)
        return loss.data

    def predict(self, inputs, batch_size=1, classes=False, disable_progbar=False):
        """Generates output predictions batch
           by batch for the input samples.

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            batch_size: integer. Number of samples per batch
            classes: boolean. Whether to return class predictions

        # Returns
            A 1D torch tensor of predictions

        #Raises
            ValueError: If the number of samples in inputs are
                        not equal
        """
        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]

        predict_dataset = dataset(inputs)
        predict_data_loader = DataLoader(
            predict_dataset, batch_size=batch_size, shuffle=False)

        return self.predict_on_generator(predict_data_loader, classes=classes)

    def predict_on_generator(self, generator, steps_per_epoch=None, classes=False):
        """Predicts the model on data generator

        # Arguments
            generator : A user created data generator or
                torch.utils.data.DataLoader. The generator should
                yield and iterable of (input_data)
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from generator before declaring one epoch
                finished and starting the next epoch. Must be specified
                for user created generator

        #Raises
            AssertionError: If generator is a user defined generator and 
                steps_per_epoch is not specified
        """
        preds = []
        if isinstance(generator, DataLoader):
            for batch in generator:
                batch_inputs = batch
                pred = self.predict_batch(
                    batch_inputs, classes=classes)
                preds.append(pred)
        else:
            steps = 0
            while steps < steps_per_epoch:
                batch_inputs = next(generator)
                pred = self.predict_batch(batch_inputs, classes=classes)
                steps += 1
                preds.append(pred)

        return torch.cat(preds)

    def predict_batch(self, inputs, classes=False):
        """Returns predictions for a single batch of samples.

        # Arguments
            inputs: A single input or a list of inputs which can be either
                numpy arrays or torch tensors
            classes: boolean. Whether to return class predictions
        # Returns
            A torch tensor of predictions
        """
        inputs = [_wrap_in_tensor(x) for x in _to_list(inputs)]

        input_batch = [Variable(x, volatile=True) for x in inputs]

        if len(input_batch) == 1:
            y = self.model(input_batch[0])
        else:
            y = self.model(input_batch)

        if classes:
            return torch.max(y.data, -1)[1]
        else:
            return y.data
