import torch
from torch import nn
import numpy as np
import flare
from flare import Trainer, dataset
from torch.utils.data import DataLoader
import pytest

x1 = np.random.rand(1000,5)
x2 = np.random.rand(1000,5)
targets = np.random.randint(0, 10, size=[1000,])

xv1 = torch.from_numpy(x1).float()
xv2 = torch.from_numpy(x2).float()
tv = torch.FloatTensor(targets).long()

train_dataset = dataset([xv1, xv2], tv)
predict_dataset = dataset([xv1, xv2])

class linear_1(nn.Module):
    
    def __init__(self):
        super(linear_1, self).__init__()
        self.dense1 = nn.Linear(5, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        return y

class linear_2(nn.Module):
    
    def __init__(self):
        super(linear_2, self).__init__()
        self.dense1 = nn.Linear(10, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, x):
        y = torch.cat(x,dim=-1)
        y = self.dense1(y)
        y = self.dense2(y)
        return y

def generator(targets=True):
    while True:
        i = 0
        bs = 32
        for i in range(0, xv1.size()[0], bs):
            if targets:
                yield [[xv1[i:i+bs], xv2[i:i+bs]], tv[i:i+bs]]
            else:
                yield [xv1[i:i+bs], xv2[i:i+bs]]

single_input_model = linear_1()
multi_input_model = linear_2()

train_generator = generator()
predict_generator = generator(targets=False)
val_data_loader = DataLoader(train_dataset, batch_size=128)
predict_data_loader = DataLoader(predict_dataset, batch_size=128)


def _get_optim(model):
    return torch.optim.Adam(model.parameters())

class TestDataset(object):

    @pytest.mark.parametrize("input_data, target_data", [
        ([xv1, xv2[:300]], tv),
        ([xv1, xv2], tv[:300])
    ])
    def test_unequal_samples(self, input_data, target_data):
        with pytest.raises(ValueError):
            t = Trainer(multi_input_model, nn.CrossEntropyLoss(), _get_optim(multi_input_model))
            t.train(input_data, target_data)


class TestTrainer(object):

    @pytest.mark.parametrize("model, data", [
        (multi_input_model, [xv1, xv2]),
        (single_input_model, xv1,),
    ])
    @pytest.mark.parametrize("validation_split, validation_data",
        [(0.0, None), (0.1, None), (0.0, None)] #### FIX THIS !!
    )
    def test_train(self, model, data, validation_split, validation_data):
        t = Trainer(model, nn.CrossEntropyLoss(), _get_optim(model))
        t.train(data, tv, validation_split=validation_split, validation_data=None, batch_size=128)

    @pytest.mark.parametrize("model, data", [
        (multi_input_model, [xv1, xv2]),
        (single_input_model, xv1),
    ])
    def test_evaluate(self, model, data):
        t = Trainer(model, nn.CrossEntropyLoss(), _get_optim(model))
        t.evaluate(data, tv, batch_size=128)

    @pytest.mark.parametrize("model, data", [
        (multi_input_model, [xv1, xv2]),
        (single_input_model, xv1)
    ])
    @pytest.mark.parametrize("classes",
        [True, False]
    )
    def test_predict(self, model, data, classes):
        t = Trainer(model, nn.CrossEntropyLoss(), _get_optim(model))
        t.predict(data, classes=classes, batch_size=128)

    @pytest.mark.parametrize("generator",
        [train_generator, val_data_loader]
    )
    @pytest.mark.parametrize("validation_data",
        [None, [(xv1, xv2), tv], train_generator, val_data_loader]
    )
    def test_train_on_generator(self, generator, validation_data):
        t = Trainer(multi_input_model, nn.CrossEntropyLoss(), _get_optim(multi_input_model))
        t.train_on_generator(generator, steps_per_epoch=31, validation_data=validation_data, validation_steps=31)

    @pytest.mark.parametrize("generator",
        [train_generator, val_data_loader]
    )
    def test_evaluate_on_generator(self, generator):
        t = Trainer(multi_input_model, nn.CrossEntropyLoss(), _get_optim(multi_input_model))
        t.evaluate_on_generator(generator, steps_per_epoch=31)

    @pytest.mark.parametrize("generator",
        [predict_generator, predict_data_loader]
    )
    def test_predict_on_generator(self, generator):
        t = Trainer(multi_input_model, nn.CrossEntropyLoss(), _get_optim(multi_input_model))
        t.predict_on_generator(generator, steps_per_epoch=31)