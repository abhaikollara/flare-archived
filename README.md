# Flare: PyTorch for everyday use

## Beware, Flare is still in developmental stages !

Flare is a utility library that enables users to train their networks on PyTorch instantly. The API was inspired from [Keras](https://github.com/keras-team/keras) deep learning framework.

Currently Flare is designed to run on PyTorch >= 0.4.0

## Guiding principles
- **"Everything should be made as simple as possible, but not simpler"**
- **Intuitiveness** Do not support complex usecases at the cost of intuitiveness of simpler tasks.


## Installation
First, clone the repo using
`https://github.com/abhaikollara/flare.git`

then `cd` to the **flare** folder and run the install command
`
cd flare && sudo python setup.py install
`

## Example
```python
import torch
from torch import nn

import flare
from flare import Trainer

input_1 = np.random.rand(10000,5)
input_2 = np.random.rand(10000,5)
targets = np.random.randint(0, 10, size=[10000,])

class linear_two_input(nn.Module):
    
    def __init__(self):
        super(linear_two_input, self).__init__()
        self.dense1 = nn.Linear(10, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, inputs):
        y = torch.cat(inputs, dim=-1)
        y = self.dense1(y)
        y = self.dense2(y)
        return y

model = linear_two_input()

t = Trainer(model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()))
t.train([input_1, input_2], targets, validation_split=0.2, batch_size=128)
```


[See MNIST example here](https://github.com/abhaikollara/flare/blob/master/examples/mnist.py)

## Why this name, Flare

 ¯\\_(ツ)_/¯
