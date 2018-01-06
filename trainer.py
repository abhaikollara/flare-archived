import torch
from torch.autograd import Variable
import numpy as np


class Trainer(object):

	def __init__(self, model, loss, metrics):
		pass


	def train(self, x, y, batch_size=None, epochs=1, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None)
		pass


	def evaluate(self, x, y, batch_size=None, metrics=['accuracy']):
		pass


	def predict(self, x, batch_size=None):
		pass


	def train_batch(self, x, y, class_weight=None):
		pass


	def evaluate_batch(self, x, y, metrics=['accuracy']):
		pass


	def predict_batch(self, x):
		pass