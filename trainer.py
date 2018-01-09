import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm



def _wrap_in_tensor(array):
	pass


def _wrap_in_variable(tensors):
	if isinstance(tensors, (list, tuple)):
		return [Variable(x) for x in tensors]
	else:
		return Variable(tensors)


def _get_optimizer(optimizer, model):
	aliases = {
		'adadelta'	: optim.Adadelta,
		'adagrad'	: optim.Adagrad,
		'adam'		: optim.Adam,
		'adamax'	: optim.Adamax,
		'asgd'		: optim.ASGD,
		'lbfgs'		: optim.LBFGS,
		'rms'		: optim.RMSprop,
		'rprop'		: optim.Rprop,
		'sgd'		: optim.SGD,
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

	def __len__(self):
		return self.inputs.size()[0]

	def __getitem__(self, idx):
		if not isinstance(self.inputs, (list, tuple)):
			return self.inputs[idx], self.targets[idx]
		else:
			return [x[idx] for x in self.inputs], self.targets[idx]


class Trainer(object):

	def __init__(self, model, loss, optimizer):
		self.model = model
		self.optimizer = _get_optimizer(optimizer, model)
		self.loss_func = loss


	def train(self, inputs, targets, batch_size=None, epochs=1, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None):
		'''
			TODO : 
			1. Validation data
			2. Shuffle
			3. Class weight
		'''
		if validation_data:
			train_dataset = dataset(inputs, targets)
			valid_dataset = None # TODO
		elif validation_split > 0.0:
			split_size = int(len(inputs) * validation_split)
			train_dataset = dataset(inputs[:-split_size], targets[:-split_size])
			valid_dataset = dataset(inputs[:-split_size], targets[:-split_size])
		else:
			train_dataset = dataset(inputs, targets)

		train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

		for epoch in range(epochs):
			print("Epoch", str(epoch), "of", str(epochs))
			for batch in tqdm(train_data_loader):
				batch_inputs, batch_targets = batch
				self.train_batch(batch_inputs, batch_targets)


	def evaluate(self, inputs, targets, batch_size=None, metrics=['accuracy']):
		pass


	def predict(self, inputs, batch_size=None):
		pass


	def train_batch(self, inputs, targets, class_weight=None):

		self.optimizer.zero_grad()
		input_batch = _wrap_in_variable(inputs)
		target_batch = _wrap_in_variable(targets)
		# TODO : Make inputs and targets accept np nd-arrayss
		y = self.model(input_batch)
		loss = self.loss_func(y, target_batch)
		loss.backward()
		self.optimizer.step()


	def evaluate_batch(self, inputs, targets, metrics=['accuracy']):
		pass


	def predict_batch(self, inputs):
		pass