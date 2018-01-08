import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def _wrap_in_tensor(array):
	pass


def _wrap_in_variable(arrays):
	pass


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



class Trainer(object):

	def __init__(self, model, loss, optimizer):
		self.model = model
		self.optimizer = _get_optimizer(optimizer, model)
		self.loss_func = loss


	def train(self, inputs, targets, batch_size=None, epochs=1, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None):
		pass




	def evaluate(self, inputs, targets, batch_size=None, metrics=['accuracy']):
		pass


	def predict(self, inputs, batch_size=None):
		pass


	def train_batch(self, inputs, targets, class_weight=None):

		self.optimizer.zero_grad()

		# TODO : Make inputs and targets accept np nd-arrayss
		y = self.model(inputs)
		loss = self.loss_func(y, targets)
		loss.backward()
		self.optimizer.step()


	def evaluate_batch(self, inputs, targets, metrics=['accuracy']):
		pass


	def predict_batch(self, inputs):
		pass