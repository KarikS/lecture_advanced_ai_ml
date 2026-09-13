---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region pycharm={"name": "#%% md\n"} -->
# Lab 1

Welcome to the first lab.

In this lab, we will finally use PyTorch as a deep learning framework (I personally also
really appreciate JAX - check it out!). We will build a small feedforward network for
sentiment classification, then look at the signs of overfitting and how to fight them
with regularization.

> **Instructor timing (2h session).** ~15 min: walkthrough of the IMDB loading and
> bag-of-words encoding below (already filled in for you, it's plumbing, not modelling).
> ~40 min: Exercise 1 - build the network, the training loop, evaluate and predict.
> ~50 min: Exercise 2 - the overfitting baseline, then weight decay and dropout.
> Early stopping and the TensorBoard demo at the end of Exercise 2 are take-home /
> instructor-demo material, not required in-session.

## Setup

These labs are designed to run on [Google Colab](https://colab.research.google.com/) -
no local Python install needed. Open this notebook via Colab's GitHub loader
(`colab.research.google.com/github/KarikS/lecture_advanced_ai_ml/blob/main/<path-to-this-notebook>`)
or File > Open notebook > GitHub tab, repo `KarikS/lecture_advanced_ai_ml`, then run
the cell below once per session. It clones this repository so the notebook can reach
`data/imdb.npz` and the other files it needs. Running the notebook locally instead
(e.g. via the `exercises/python/` venv described in that folder's README) works
unchanged - the cell below is a no-op there.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    import os
    if not os.path.exists('lecture_advanced_ai_ml'):
        !git clone https://github.com/KarikS/lecture_advanced_ai_ml.git
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
from operator import itemgetter
import json
from pathlib import Path
import random
from typing import Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import RMSprop
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats('png', 'pdf');
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1

In this exercise, we will learn the basic usage of PyTorch, a popular deep learning
library. We already utilized PyTorch in the exercises before but only to construct
matrices and perform operations on them.

The network we'll use here is tiny and runs fine on Colab's default CPU runtime - no
need to switch on a GPU for this particular lab. (Later labs will actually need one;
when they do, get it via "Runtime" -> "Change runtime type" -> GPU.)

### Loading and preparing the dataset

The dataset that we will be working with is the IMDB dataset. It contains 50,000
reviews that are highly polarized, that is, they are unambiguously either 'positive'
or 'negative'. The training and test sets contain 25,000 reviews each, and in both
sets half of the reviews are positive and half are negative.

The cells below are already complete - this part is tokenization plumbing, not deep
learning, so we'll just walk through it together rather than have you write it.
We load the raw dataset from a local `.npz` file (`data/imdb.npz`, shipped in this
repo) instead of `keras.datasets.imdb`, so that nothing here requires installing
TensorFlow/Keras. It is the exact same data, in the exact same encoding.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
DATA_DIR = (
    Path('lecture_advanced_ai_ml/exercises/python/data') if IN_COLAB else Path('../data')
)

def load_imdb(data_dir: Path) -> Tuple[Tuple[list, list], Tuple[list, list]]:
    """Load the raw IMDB review/label arrays from the local .npz file."""
    with np.load(data_dir / 'imdb.npz', allow_pickle=True) as f:
        train_x = [list(seq) for seq in f['x_train']]
        train_y = list(f['y_train'])
        test_x = [list(seq) for seq in f['x_test']]
        test_y = list(f['y_test'])
    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = load_imdb(DATA_DIR)

with open(DATA_DIR / 'imdb_word_index.json') as f:
    word2enc = json.load(f)
enc2word = {v: k for k, v in word2enc.items()}
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's check a random review!
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
print(train_x[random.randint(0, len(train_x) - 1)])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
This is obviously just the numeric encoding of the sentences. To get something that is
in fact readable we need to use the `enc2word` mapping. The `+3` offset accounts for the
reserved tokens 0 (padding), 1 (start of sequence) and 2 (unknown/out-of-vocabulary).
<!-- #endregion -->

```python
print(" ".join([enc2word[enc+3] for enc in train_x[random.randint(0, len(train_x) - 1)]]))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Before we feed the reviews to the network, we need to convert them from sequences of
integers to "bag of words" vectors. For example, turning the sequence (3,5,9)
into a 10 dimensional vector gives us (0,0,1,0,1,0,0,0,1,0),
which has a 1 in the positions 3, 5 and 9 and zeros everywhere else.

We only keep the 10,000 most common words (word indices are already ranked by
frequency in the raw data, so this is just a cutoff on the index), which will be the
size of the input vector.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
def get_index_vector(sequence: list[int], size: int = 10000) -> list[int]:
    output_vec = [0 for _ in range(size)]
    for index in sequence:
        if index is None:
            continue
        elif index <= size - 1:
            output_vec[index] = 1
    return output_vec

train_x = [get_index_vector(count_vector) for count_vector in train_x]
test_x = [get_index_vector(count_vector) for count_vector in test_x]
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We now have everything ready to built a Pytorch `Dataset` object,
which is recommended for the training process.

We also define `device`. This indicates the location where we would like to process data.
By default tensors are on the CPU. If we have a GPU, we could set `device` to `cuda` to
utilize GPU power. If we have set a device, we can then push a tensor to the desired location
by calling `<tensor>.to(device)`. Tensors can also be created directly on the device
by specifying the `device` argument on tensor initialization.
In this setting device is set to `cuda` if a GPU is
available, otherwise we'll just use the cpu.

Pushing the whole dataset to GPU is often not a possibility due to memory constraints,
but in this exercise the small vectorized IMDB data will only consume around 3GB VRAM.
We could also save memory by reducing the size of the input vector or using sparse tensors.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMDBDataset(Dataset):
    def __init__(self, data: Union[list, Tuple], labels: list, device: torch.device):
        self.data = torch.tensor(data, dtype=torch.float, device=device)
        self.labels = torch.tensor(labels, dtype=torch.float, device=device)

    # We don't need this method necessarily,
    # but it is usually good to have direct access to the dimensions of the dataset.
    @property
    def shape(self) -> Tuple:
        return self.data.shape

    # The `__len__` method should return the number of samples in the dataset.
    # This will later on be used automatically by the data loader.
    def __len__(self) -> int:
        return len(self.labels)

    # Each Dataset needs to implement the `__get_item__` method.
    # The method gets an index and should return the corresponding items.
    # For example index = 5 should return the 5th review and its matching label.
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.labels[idx]
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let’s also create a validation set to monitor the (generalization) performance of the model during training,
by randomly taking the 10,000 samples of the training data and the corresponding labels.
The new training set should contain only the remaining 15,000 samples.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Get random indices for training and validation split
shuffled_indices = list(range(len(train_x)))
random.seed(42)
random.shuffle(shuffled_indices)

train_idxs = shuffled_indices[:15000]
val_idxs = shuffled_indices[15000:]

# Plain python does not know multi index selection
# We can still use `itemgetter` from the `operator` module to achieve what we want.
val_x = itemgetter(*val_idxs)(train_x)
val_y = itemgetter(*val_idxs)(train_y)

train_x = itemgetter(*train_idxs)(train_x)
train_y = itemgetter(*train_idxs)(train_y)

# We can now finally initialize our PyTorch datasets.
train_dataset = IMDBDataset(train_x, train_y, device)
val_dataset = IMDBDataset(val_x, val_y, device)
test_dataset = IMDBDataset(test_x, test_y, device)

print('Training\t Shape: {}'.format(train_dataset.shape))
print('Validation\tShape: {}'.format(val_dataset.shape))
print('Test\t\tShape: {}'.format(test_dataset.shape))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Our data is now ready to be fed to a neural network.

Let's remove the preprocessed data lists to free some RAM:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
del train_x
del train_y
del val_x
del val_y
del test_x
del test_y
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Building the network
When deciding on an architecture for a neural network with fully connected layers,
the two key things to consider are:

 - The number of hidden layers in the network
 - The number of neurons in each of the hidden layers

Increasing the depth of the network (that is, adding layers) or increasing the number
of neurons in a given layer will increase the capacity of the network and allow it
to learn more complex non-linear decision boundaries.
However, making the network too large can lead to overfitting.

In practice, deciding on an architecture is an iterative process where
many different networks are trained in order to find a good setting for the
hyperparameters. For this exercise, however, we will use a simple feedforward neural
network with two fully-connected hidden layers consisting of 16 neurons each,
and a single output neuron which outputs the probability of the review being 'positive'.

In PyTorch the `nn` module holds all building blocks that we need.
The `Sequential` module allows us to specify a neural network layer by layer.
A fully connected layer is available as `Linear`.
Simiarly we can import the activation functions as `ReLU` and `Sigmoid`.
We need to push the model to our `device` as well.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
input_size = train_dataset.shape[1]

model = nn.Sequential(
    nn.Linear(in_features=input_size, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=16),
    nn.ReLU(),
    #!TAG HWBEGIN
    #!MSG TODO: Add the output neuron and activation.
    #!MSG Think about which activation turns a raw score into something you can
    #!MSG interpret as a probability of the review being 'positive'.
    nn.Linear(in_features=16, out_features=1),
    nn.Sigmoid()
    #!TAG HWEND
).to(device)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
So, we've built our model. Before we can train the network, however, we must specify:

1. The loss function to use (mean squared error, cross entropy, etc) (Info: [here](https://pytorch.org/docs/stable/nn.html#loss-functions))
2. The optimizer (SGD, Adam, RMSProp, etc.) (Info: [here](https://pytorch.org/docs/stable/optim.html))
3. Any metrics (such as accuracy) used to measure the performance of the model
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

loss = (
    #!TAG HWBEGIN
    #!MSG TODO: Add the appropriate loss for binary classification.
    #!MSG Hint: the network's output is already a probability (see the Sigmoid above).
    nn.BCELoss()
    #!TAG HWEND
)

optimizer = (
    #!TAG HWBEGIN
    #!MSG TODO: Add the RMSprop optimizer.
    #!MSG Note: You also need the specify the parameters we want to optimize.
    #!MSG You get these by calling `model.parameters()`.
    RMSprop(model.parameters())
    #!TAG HWEND
)

def get_accuracy(prediction: Tensor, target: Tensor) -> float:
    return float(torch.sum(((prediction > 0.5) == target).float()) / len(prediction))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Fit
We wish to train the network for 20 epochs with batches of size 512.
In contrast to e.g. Keras, in the basic PyTorch approach we need to implement our
training routine mostly from scratch. Luckily PyTorch provides a `DataLoader` utility,
which allows to easily sample from our `Datasets`.
On the downside we still need to manage metrics on our own.

The `train` function below implements one version of a training loop.
The outer `for` loop is for the number of training epochs. The inner `for` loop
iterates over the whole dataset with the help of the `Dataloader`.
The output of the dataloader is dependend on batch size and the definition of the
provided `Dataset`. In our case we specified in the `__get_item__` function of
`IMDBDataset` that a  tuple with one training sample and label should be returned.
The dataloader does batching and collating automatically. This means in the backend
the loader utilizes our `__get_item__` method but presents us batched results.
In other words, for a batch size of 512 the loader gives us a tuple of a 512 x 10000
sample matrix and a label vector of length 512, which we directly unpack into `x` and `y`.

The actual training logic is straightforward. First, we need to do a forward pass and
compute the loss. The optimization itself follows three steps.

1. Make sure all gradients of the parameter tensors are zeroed or None (`optim.zero_grad()`)
2. Backpropagate the error (`loss.backward()`)
3. Apply the gradients using the optimizer (`optim.step`)

The validation loop follows the exact same principles, but obviously we don't do any
optimization steps. The `torch.no_grad()` context manager implies that no gradients
are calculated and no results are stashed on the computation. This makes a significant
difference in speed in larger models.

You may notice that we have set the model in training (`model.train()`) and evaluation
(`model.eval()`) mode. For our simple model this doesn't make a difference, but for some
layers like `Dropout` or `BatchNormalization` this setting triggers different policies.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def train(
    model: nn.Module,
    loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    epochs: int
) -> dict:

    # Define a dict with room for metrics that will be populated during training.
    metrics: dict = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    # The loader allows to shuffle the training data on the fly.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for ep in range(1, epochs + 1):

        batch_losses = []
        predictions = []
        targets = []

        for batch_idx, (x, y) in enumerate(train_loader):

            ############################################
            # TRAINING LOGIC
            ############################################
            # Set the model to training mode
            model.train()

            # Forward pass through the model
            y_hat = model(x).squeeze()
            # Obtain the loss
            batch_loss = loss(y_hat, y)

            #!TAG HWBEGIN
            #!MSG TODO: Add backpropagation of the loss and apply the gradients via the optimizer
            # Set all parameter gradients to zero
            optimizer.zero_grad()
            # Backpropagate the error
            batch_loss.backward()

            # Apply gradients
            optimizer.step()
            #!TAG HWEND

            ############################################

            batch_losses.append(batch_loss)
            predictions.append(y_hat.detach())
            targets.append(y)


        ep_train_loss = float(torch.mean(torch.stack(batch_losses)))
        ep_train_acc = get_accuracy(torch.cat(predictions), torch.cat(targets))

        batch_losses.clear()
        predictions.clear()
        targets.clear()

        for batch_idx, (x, y) in enumerate(val_loader):

            ############################################
            # VALIDATION LOGIC
            ############################################
            # Set the model to evaluation mode
            model.eval()

            with torch.no_grad():

                #!TAG HWBEGIN
                #!MSG TODO: Do a forward pass and get the batch loss
                # Forward pass through the model
                y_hat = model(x).squeeze()
                # Obtain the loss
                batch_loss = loss(y_hat, y)
                #!TAG HWEND
            ############################################

            batch_losses.append(batch_loss)
            predictions.append(y_hat.detach())
            targets.append(y)

        ep_val_loss = float(torch.mean(torch.stack(batch_losses)))
        ep_val_acc = get_accuracy(torch.cat(predictions), torch.cat(targets))

        metrics['train_loss'].append(ep_train_loss)
        metrics['train_acc'].append(ep_train_acc)
        metrics['val_loss'].append(ep_val_loss)
        metrics['val_acc'].append(ep_val_acc)

        print('EPOCH:\t{:5}\tTRAIN LOSS:\t{:.3f}\tTRAIN ACCURACY:\t{:.2f}'
              '\tVAL LOSS:\t {:.5f}\tVAL ACCURACY:\t {:.2f}'
              .format(ep, ep_train_loss, ep_train_acc, ep_val_loss, ep_val_acc), end='\r')

    return metrics

metrics = train(
    model=model,
    loss=loss,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    batch_size=512
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's visualize the training progress.
We can utilize the `metrics` that are returned from our `train` method.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def get_training_progress_plot(
        train_losses: list[float],
        train_accs: list[float],
        val_losses: list[float],
        val_accs: list[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2))

    ax1.set_title('Loss')
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.legend()
    #ax1.set_yscale('log')

    ax2.set_title('Accuracy')
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.legend()

get_training_progress_plot(
    metrics['train_loss'],
    metrics['train_acc'],
    metrics['val_loss'],
    metrics['val_acc'],
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As expected, the training loss decreases with each epoch (and training accuracy increases).
However, the validation loss decreases initially and then begins to increase after
the first few epochs epochs. Therefore, the network has overfit.

### Evaluate
Let's evaluate the performance of the model on the test set:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def evaluate(model: nn.Module, test_dataset: Dataset, batch_size: int = 512) -> dict:
    batch_losses: list = []
    predictions: list = []
    targets: list = []

    for x, y in DataLoader(test_dataset, batch_size):
        #!TAG HWBEGIN
        #!MSG TODO: Do a forward pass and get the batch loss
        model.eval()
        with torch.no_grad():
            y_hat = model(x).squeeze()
            batch_loss = loss(y_hat, y)

        #!TAG HWEND
        ############################################

        batch_losses.append(batch_loss)
        predictions.append(y_hat.detach())
        targets.append(y)

    eval_loss = float(torch.mean(torch.stack(batch_losses)))
    eval_acc = get_accuracy(torch.cat(predictions), torch.cat(targets))

    return {'test_loss': eval_loss, 'test_acc': eval_acc}

eval_metrics = evaluate(model, test_dataset)
print(eval_metrics)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Our simple model does reasonably well. It achieves an accuracy of around 85-88%.

### Predict
Finally, to generate the likelihood of the reviews being positive, we only need to
forward data through our fitted model:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
sample_x = test_dataset.data[:10]
sample_y = test_dataset.labels[:10]

predictions = (
    #!TAG HWBEGIN
    #!MSG TODO: Forward `sample_x` through the trained model to get its predictions.
    model(sample_x)
    #!TAG HWEND
)

for i in range(len(sample_x)):
    print('{} | TRUE: {} | PRED {:.2e}'.format(i, int(sample_y[i]), float(predictions[i])))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now play around with the code by adding and deleting layers, changing the hidden activation, optimizer, learning rate, batch-size, etc.

### Conclusion
Here's what you should take away from this example:

 - You usually need to do quite a bit of preprocessing on your raw data in order to be
able to feed it -- as tensors -- into a neural network. Encoding the words as BOW vectors
is only the tip of the iceberg.
 - Stacks of dense layers with `ReLU` activations can solve a wide range of problems
(including sentiment classification), and you'll likely use them frequently.
 - As they get better on their training data, neural networks eventually start
_overfitting_ and end up obtaining increasingly worse results on data they have never
seen before. Be sure to always monitor performance on data that is outside of the training set.


## Exercise 2
In this exercise, we will look at a couple of different methods to regularize a neural
network in order to prevent overfitting.

Plotting the validation loss is a simple way to determine whether the network has
overfit. During the first few epochs of training, both the training loss and the
validation loss tend to decrease in tandem but after a certain point, the validation
loss starts to increase while the training loss continues to decrease.
It is at this point that the network begins to overfit.

### The overfitting baseline

We re-create the same architecture from Exercise 1 as our baseline model for
comparison, and train it the same way.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
input_size = train_dataset.shape[1]

original_model = (
    #!TAG HWBEGIN
    #!MSG TODO: Create the network according to the specifications above and push it to `device`.
        nn.Sequential(
        nn.Linear(in_features=input_size, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=1),
        nn.Sigmoid()
    ).to(device)
    #!TAG HWEND
)

print('Training baseline model...')
original_model_metrics = train(
    #!TAG HWBEGIN
    #!MSG TODO: Fill in the correct parameters for the function
    model=original_model,
    loss=nn.BCELoss(),
    optimizer=RMSprop(original_model.parameters()),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    batch_size=512
    #!TAG HWEND
)

get_training_progress_plot(
    original_model_metrics['train_loss'],
    original_model_metrics['train_acc'],
    original_model_metrics['val_loss'],
    original_model_metrics['val_acc'],
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As before, the training loss keeps dropping while the validation loss turns back
upward - our baseline overfits. This is the reference we'll compare every
regularized version of the model against.

We'll also define a small helper to compare the validation loss of two runs directly.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def compare_losses_plot(
        first_losses: list[float],
        second_losses: list[float],
        first_loss_label: str,
        second_loss_label: str
) -> None:
    plt.plot(first_losses, label=first_loss_label)
    plt.plot(second_losses, label=second_loss_label)
    plt.legend()
    plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Adding weight regularization

Regularizing a model in PyTorch can be done over the additional `weight_decay`
argument in the optimizer function. By setting a coefficient there, the linked
parameters will have a L2 penalty.

Redeclare the original model, then set up an optimizer with a `weight_decay`
coefficient of 0.001 and fit it.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
regularized_model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=1),
        nn.Sigmoid()
).to(device)

regularized_optimizer = (
    #!TAG HWBEGIN
    #!MSG TODO: Create an RMSprop optimizer for `regularized_model` with weight_decay=0.001.
    RMSprop(regularized_model.parameters(), weight_decay=0.001)
    #!TAG HWEND
)

regularized_model_metrics = train(
    model=regularized_model,
    loss=nn.BCELoss(),
    optimizer=regularized_optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    batch_size=512
)

compare_losses_plot(
    original_model_metrics['val_loss'],
    regularized_model_metrics['val_loss'],
    'original',
    'weight decay'
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As you can see, the regularized model does not overfit as much,
even though both models have the same number of parameters.
Feel free to play with the regularization strength to get a feel on how different
settings affect learning. When is regularization preventing the network from
learning anything at all? When is regularization so weak it does not make a difference?

### Dropout regularization
Dropout is a very popular technique to regularize neural nets.
It works by randomly turning off (or "dropping out")  the input/hidden neurons in a
network. This means that every neuron is trained on a different set of examples.
Note that dropout is, in most cases, only used during training time.
At test time, all units are used with their activations scaled down by the dropout rate
to account for the fact that all neurons were used for the prediction.
Normally, dropout is not applied to the inputs.

In torch, dropout is implemented as its own separate layer (`Dropout`) that takes as
input the probability to _drop_ units. To apply dropout to a layer, place a `Dropout` after
it while stacking layers. The dropout will be ignored if the model is in `eval` mode.
Luckily, we already set the correct modes for training and evaluation in our `train`
function.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
dropout_model = nn.Sequential(
        #!TAG HWBEGIN
        #!MSG TODO: Keep the structure of the original model,
        #!MSG but add dropout after the hidden layers with prob=0.5
        nn.Linear(in_features=input_size, out_features=16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=16, out_features=16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=16, out_features=1),
        nn.Sigmoid()
        #!TAG HWEND
).to(device)

dropout_model_metrics = train(
    model=dropout_model,
    loss=nn.BCELoss(),
    optimizer=RMSprop(dropout_model.parameters()),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    batch_size=512
)

compare_losses_plot(
    original_model_metrics['val_loss'],
    dropout_model_metrics['val_loss'],
    'original',
    'dropout'
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Once again, we see a marked improvement over the baseline. Weight decay and dropout
attack overfitting from different angles - weight decay shrinks the weights directly,
dropout prevents co-adaptation between neurons - but here they help by a similar
amount. In practice you'd often combine both, and tune their strength as
hyperparameters.

### Early Stopping (take-home / instructor demo - not required in the 2h session)

Previously, we were training the network and checking _after training_ when it started
to overfit. But another very popular method to regularize a network is to stop
training earlier than the specified number of epochs, by checking when the validation
loss starts to increase.

There are no out of box utilities to achieve this kind of behavior, so we adjust
the training loop accordingly, adding a `patience` argument: how many epochs to wait
for an improvement of the validation loss before interrupting training.

NOTE: We only implement a very naive method of early stopping. Usually you would
need to roll back the weights to the epoch which had the last improvement (early
stopping epoch minus patience). However, checkpointing and caching weights is out
of scope of this exercise.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def train_with_early_stopping(
    model: nn.Module,
    loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    epochs: int,
    patience: int = 2,
) -> dict:

    metrics: dict = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    # Track how often in a row no improvements happen
    early_stopping_strikes = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for ep in range(1, epochs + 1):

        batch_losses = []
        predictions = []
        targets = []

        for batch_idx, (x, y) in enumerate(train_loader):
            model.train()

            y_hat = model(x).squeeze()
            batch_loss = loss(y_hat, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss)
            predictions.append(y_hat.detach())
            targets.append(y)

        ep_train_loss = float(torch.mean(torch.stack(batch_losses)))
        ep_train_acc = get_accuracy(torch.cat(predictions), torch.cat(targets))

        batch_losses.clear()
        predictions.clear()
        targets.clear()

        for batch_idx, (x, y) in enumerate(val_loader):
            model.eval()

            with torch.no_grad():
                y_hat = model(x).squeeze()
                batch_loss = loss(y_hat, y)

            batch_losses.append(batch_loss)
            predictions.append(y_hat.detach())
            targets.append(y)

        ep_val_loss = float(torch.mean(torch.stack(batch_losses)))
        ep_val_acc = get_accuracy(torch.cat(predictions), torch.cat(targets))

        metrics['train_loss'].append(ep_train_loss)
        metrics['train_acc'].append(ep_train_acc)
        metrics['val_loss'].append(ep_val_loss)
        metrics['val_acc'].append(ep_val_acc)

        print('EPOCH:\t{:5}\tTRAIN LOSS:\t{:.3f}\tTRAIN ACCURACY:\t{:.2f}'
              '\tVAL LOSS:\t {:.5f}\tVAL ACCURACY:\t {:.2f}'
              .format(ep, ep_train_loss, ep_train_acc, ep_val_loss, ep_val_acc), end='\r')

        ###############################################################
        # EARLY STOPPING
        ###############################################################
        if ep > 1:
            if metrics['val_loss'][-2] <= metrics['val_loss'][-1]:
                early_stopping_strikes += 1
            else:
                early_stopping_strikes = 0

            if early_stopping_strikes >= patience:
                break

    return metrics

early_dropout_model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=16, out_features=16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=16, out_features=1),
        nn.Sigmoid()
).to(device)

early_dropout_model_metrics = train_with_early_stopping(
    model=early_dropout_model,
    loss=nn.BCELoss(),
    optimizer=RMSprop(early_dropout_model.parameters()),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    batch_size=512,
    patience=2
)

compare_losses_plot(
    early_dropout_model_metrics['val_loss'],
    dropout_model_metrics['val_loss'],
    'early stopping',
    'dropout'
)

eval_metrics = evaluate(early_dropout_model, test_dataset)
print(eval_metrics)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As you can see, the early stopping callback worked, the model was trained for only a
few epochs, and the resulting test loss is close to the lowest loss in the graph.

The take-home message for this exercise is: large neural networks can easily overfit,
especially with small training sets. This means that the network learns spurious patterns
that are present in the training data and, therefore, fails to generalize to unseen examples.
In such a scenario, your options are:

 1. Get more training data
 2. Reduce the size of the network
 3. Regularize the network

### From hand-rolled plots to experiment tracking (instructor demo)

Notice what we just did four times: train a model, pull `train_loss`/`val_loss` out
of a dict, call `plt.plot`, eyeball the curves. That's manual experiment tracking,
and it does not scale past a handful of runs - you won't want to do this by hand once
you're sweeping hyperparameters or comparing a dozen architectures.

`torch.utils.tensorboard.SummaryWriter` gives you the same comparison for free, with
interactive, zoomable curves, one line of logging per metric. We log the runs we
already trained above into TensorBoard - in your own training loops going forward,
you'd call `writer.add_scalar` inside the epoch loop instead of appending to a dict.
We'll build on this - alongside MLflow and Weights & Biases - in Session 10.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

runs = {
    'baseline': original_model_metrics,
    'weight_decay': regularized_model_metrics,
    'dropout': dropout_model_metrics,
}

for run_name, run_metrics in runs.items():
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    for epoch, (train_loss, val_loss) in enumerate(
        zip(run_metrics['train_loss'], run_metrics['val_loss'])
    ):
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.close()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now launch TensorBoard (in a notebook: `%load_ext tensorboard` then
`%tensorboard --logdir runs`; from a terminal: `tensorboard --logdir runs`) and
compare all three runs' loss curves in one interactive view - no `compare_losses_plot`
calls required.
<!-- #endregion -->
