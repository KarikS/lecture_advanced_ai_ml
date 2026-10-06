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
# Lab 6

> Adapted from the UvA Deep Learning Tutorials' "Vision Transformers" notebook
> (`phlippe/uvadlc_notebooks`, Tutorial 15, MIT license). The source trains a larger ViT
> on the full CIFAR-10 training set for 180 epochs on a GPU; this lab uses a smaller ViT,
> a 4,000-image subset and 15 epochs so that it runs on a CPU within a lab session.

Session 10 argued that Vision Transformers lack the inductive biases (locality,
translation equivariance) that convolutions have built in, and so need more data to
reach the same performance a CNN gets "for free". This lab makes that argument concrete:
we train a small ViT and a small CNN on the *exact same* CIFAR-10 subset, for the same
number of epochs, and compare.

> **Instructor timing (2h session).** ~10 min: intro (this page) and setup - the CIFAR-10
> download can be slow on a bad connection, so start this early. ~30 min: Exercise 1 -
> implement patch embedding and assemble the ViT. ~45 min: Exercise 2 - write the
> training and evaluation code, train the ViT and a CNN baseline under an identical,
> small data/epoch budget, and compare (training runs for several minutes - discuss the
> questions while it runs). ~25 min: Exercise 3 - extract and visualize the `[CLS]`
> token's attention in every layer. ~10 min buffer.

## Setup

These labs are designed to run on [Google Colab](https://colab.research.google.com/) -
no local Python install needed. Open this notebook via Colab's GitHub loader
(`colab.research.google.com/github/KarikS/lecture_advanced_ai_ml/blob/main/<path-to-this-notebook>`)
or File > Open notebook > GitHub tab, repo `KarikS/lecture_advanced_ai_ml`, then run
the cell below once per session. Running the notebook locally instead (e.g. via the
`exercises/python/` venv described in that folder's README) works unchanged - the cell
below is a no-op there. No GPU is needed - everything below is deliberately scaled to
train on CPU. Exercise 2's two training runs take about 5 minutes on a 4-core laptop CPU,
and can take noticeably longer on Colab's free tier.
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
import time

import matplotlib.pyplot as plt
import torch
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

set_matplotlib_formats('png', 'pdf')
torch.manual_seed(0)

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1: Patch Embedding and the ViT Architecture

Recall from Session 10: a ViT splits an image into fixed-size patches, linearly projects
each patch into a token embedding, prepends a learnable `[CLS]` token, adds a learned
position embedding to every token, and then runs an ordinary Transformer encoder
(exactly the encoder from Session 7) over the resulting sequence. The final layer's
`[CLS]` token representation is used for classification - the same mechanism BERT's
`[CLS]` token uses (Session 8).

CIFAR-10 images are $32\times32$. With a patch size of 4, that gives
$(32/4)^2=64$ patches per image, plus the `[CLS]` token: 65 tokens total, each
`embed_dim`-dimensional - short enough sequences that even a CPU can process a batch
quickly.

The cleanest way to implement patch embedding is a single strided convolution: a
`Conv2d` with `kernel_size=stride=patch_size` applied to a $32\times32\times3$ image
produces one `embed_dim`-dimensional vector per non-overlapping $4\times4$ patch, all in
one call - exactly the linear patch projection from Session 10, just implemented via a
convolution instead of an explicit reshape-then-linear (the two are mathematically
identical here, since the "kernel" never slides over overlapping regions).
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        #!TAG HWBEGIN
        #!MSG TODO: Define a Conv2d that maps `in_chans` input channels to `embed_dim`
        #!MSG output channels, with kernel_size and stride both equal to patch_size.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #!TAG HWEND

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (batch, embed_dim, H / patch_size, W / patch_size)
        #!TAG HWBEGIN
        #!MSG TODO: Flatten the two spatial dimensions into one sequence dimension, then
        #!MSG move the embedding dimension last, giving shape (batch, num_patches, embed_dim).
        #!MSG Hint: .flatten(2) then .transpose(1, 2).
        return x.flatten(2).transpose(1, 2)
        #!TAG HWEND
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
# Sanity check: a batch of 8 CIFAR-sized images should become 8 sequences of 64 patches.
patch_embed = PatchEmbed(patch_size=4, in_chans=3, embed_dim=32)
dummy_images = torch.randn(8, 3, 32, 32)
dummy_patches = patch_embed(dummy_images)
print(dummy_patches.shape)
assert dummy_patches.shape == (8, 64, 32)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now the full ViT: prepend a learnable `[CLS]` token, add a learned position embedding,
run PyTorch's built-in `nn.TransformerEncoder` (the same building block Session 7
derived from scratch), and classify from the final `[CLS]` representation.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
class ViT(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, embed_dim: int = 64,
                 depth: int = 4, heads: int = 4, mlp_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        #!TAG HWBEGIN
        #!MSG TODO: Create the learnable [CLS] token, shape (1, 1, embed_dim), and the
        #!MSG learned position embedding, shape (1, num_patches + 1, embed_dim). Both
        #!MSG should be nn.Parameter so they are optimized during training.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        #!TAG HWEND
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)

        #!TAG HWBEGIN
        #!MSG TODO: Expand cls_token to this batch size and concatenate it in front of
        #!MSG the patch tokens (along the sequence dimension), then add the position
        #!MSG embedding.
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        #!TAG HWEND

        x = self.encoder(x)
        #!TAG HWBEGIN
        #!MSG TODO: Take the first token (the [CLS] token) from the encoder output,
        #!MSG normalize it, and pass it through the classification head.
        return self.head(self.norm(x[:, 0]))
        #!TAG HWEND
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
vit_sanity = ViT()
dummy_logits = vit_sanity(dummy_images)
print(dummy_logits.shape)
assert dummy_logits.shape == (8, 10)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 2: ViT vs. CNN, Same Data, Same Budget

We now load CIFAR-10. Full CIFAR-10 is 50,000 training images - more than we can afford
several training runs over on CPU within a lab session. We deliberately train on a
*small* subset (a few thousand images) for a *few* epochs - this is the whole point:
we want to see how each architecture behaves when data and training time are both
scarce, not to reach the source tutorial's 180-epoch, full-dataset accuracy.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
transform = Compose([ToTensor(), Normalize(CIFAR_MEAN, CIFAR_STD)])
train_full = CIFAR10(root='.data', train=True, download=True, transform=transform)
test_full = CIFAR10(root='.data', train=False, download=True, transform=transform)

N_TRAIN = 4000
N_TEST = 1000

# Shuffle before subsetting: some datasets store examples grouped by class. CIFAR-10
# does not, but it is good practice never to assume a raw ordering is safe to slice.
generator = torch.Generator().manual_seed(0)
train_subset_idx = torch.randperm(len(train_full), generator=generator)[:N_TRAIN]
test_subset_idx = torch.randperm(len(test_full), generator=generator)[:N_TEST]

train_dataset = Subset(train_full, train_subset_idx.tolist())
test_dataset = Subset(test_full, test_subset_idx.tolist())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)
# The same training images again, unshuffled - only used to measure training accuracy.
train_eval_loader = DataLoader(train_dataset, batch_size=256, num_workers=2)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Here is a small CNN baseline - three convolution/pooling blocks followed by two linear
layers, the same recipe Lab 3 used for MNIST, just scaled up slightly for CIFAR-10's
larger, 3-channel images.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            #!TAG HWBEGIN
            #!MSG TODO: Count how many images in this batch are classified correctly
            #!MSG (predicted class = argmax over the logits) and add that to n_correct;
            #!MSG add the batch size to n_total.
            logits = model(x)
            n_correct += int((logits.argmax(dim=1) == y).sum())
            n_total += len(y)
            #!TAG HWEND
    return n_correct / n_total


def train_and_evaluate(model: nn.Module, epochs: int, lr: float = 1e-3) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    test_accuracies = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            #!TAG HWBEGIN
            #!MSG TODO: One optimization step on this batch: forward pass, cross-entropy
            #!MSG loss with loss_fn, backward pass, optimizer step. Call the loss `loss`.
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            #!TAG HWEND
            total_loss += float(loss) * len(y)
            n += len(y)

        test_acc = accuracy(model, test_loader)
        test_accuracies.append(test_acc)
        print('EPOCH:\t{:2}\tTRAIN LOSS:\t{:.3f}\tTEST ACCURACY:\t{:.3f}\tELAPSED:\t{:.1f}s'
              .format(epoch, total_loss / n, test_acc, time.time() - t0))
    return test_accuracies

# Both models get the same data, epochs, optimizer and learning rate. 1e-3 was the better
# of the two learning rates tested (3e-4, 1e-3) for *both* models under this budget.
EPOCHS = 15

print('Training CNN baseline...')
torch.manual_seed(0)
cnn_model = SmallCNN()
cnn_accuracies = train_and_evaluate(cnn_model, epochs=EPOCHS)

print('Training ViT...')
torch.manual_seed(0)
vit_model = ViT()
vit_accuracies = train_and_evaluate(vit_model, epochs=EPOCHS)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.plot(range(1, EPOCHS + 1), cnn_accuracies, marker='o', label='CNN')
plt.plot(range(1, EPOCHS + 1), vit_accuracies, marker='o', label='ViT')
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.legend()
plt.show()

for name, model in [('CNN', cnn_model), ('ViT', vit_model)]:
    print('{}: train accuracy {:.3f}, test accuracy {:.3f}'
          .format(name, accuracy(model, train_eval_loader), accuracy(model, test_loader)))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
On this small a data/epoch budget, the CNN should end up clearly ahead of the ViT on
test accuracy - when this lab was tested over three random seeds, the CNN reached
0.54-0.58 and the ViT 0.44-0.46. The CNN's convolutional inductive bias (locality,
translation equivariance) gives it a head start the ViT has to *learn* from data,
exactly as Session 10 argued. The source tutorial shows how much it takes to close
that gap from scratch: with the full 50,000-image training set and 180 epochs on a GPU,
its ViT reaches 75.6% test accuracy - while the CNNs in the same tutorial series reach
around 90% on CIFAR-10.

**Question:** Now look at the *training* accuracies printed above, not just the test
accuracies. Is the ViT behind because it overfits the small training set?
#!TAG HWBEGIN

**Answer:** No - its training accuracy is far lower too (in testing: ViT around
0.57-0.62, CNN around 0.77-0.79). Under this budget the ViT is also much slower to fit the training data
at all, so part of the test gap is simply that. The CNN is actually the one with the
bigger train-test gap, and it still wins on test accuracy. There is a real
generalization difference on top of that: when the lab was tested with the CNN at a
lower learning rate (3e-4), its training accuracy (0.54-0.56) was close to the ViT's -
and at that matched fit, the CNN still tested 4-6 points higher. Both effects are
consequences of the same thing: the CNN's built-in assumptions make useful image
features easier to find *and* more likely to transfer to new images.
#!TAG HWEND

**Question:** If you had a much bigger compute and data budget, would you expect this
gap to shrink, stay the same, or reverse?
#!TAG HWBEGIN

**Answer:** Shrink, and eventually reverse. Session 10's point: up to a certain dataset
size, CNNs beat ViTs because of their built-in assumptions, but the Transformer scales
well enough to make up for the missing inductive biases with massive amounts of data.
Beyond that point, not being restricted to the patterns convolutions can express
becomes an advantage.
#!TAG HWEND

## Exercise 3: What Does the `[CLS]` Token Actually Attend To?

The ViT's final classification relies entirely on the `[CLS]` token's representation
after the last encoder layer. We can inspect, in each of the 4 encoder layers, the
attention weights from the `[CLS]` token to every patch, to see where it collects its
information from.

`nn.TransformerEncoderLayer` doesn't expose its attention weights by default - we need
to call its internal `self_attn` module directly with `need_weights=True`. Our encoder
layers apply attention to their input *before* any normalization (PyTorch's default,
post-norm layout), so the input to layer $l$'s `self_attn` is simply the output of
layer $l-1$.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def get_cls_attention(model: ViT, x: Tensor, layer: int) -> Tensor:
    """Attention from [CLS] to the 64 patches in encoder layer `layer`, shape (batch, 64)."""
    model.eval()
    with torch.no_grad():
        layers = model.encoder.layers
        #!TAG HWBEGIN
        #!MSG TODO:
        #!MSG 1. Rebuild the encoder's input exactly as ViT.forward does: patch embedding,
        #!MSG    [CLS] token in front, plus the position embedding.
        #!MSG 2. Run it through layers[0], ..., layers[layer - 1] to get the input of
        #!MSG    layers[layer].
        #!MSG 3. Call layers[layer].self_attn(h, h, h, need_weights=True,
        #!MSG    average_attn_weights=True). It returns (output, attention weights); the
        #!MSG    weights have shape (batch, seq_len, seq_len), averaged over the heads.
        #!MSG 4. Row 0 is the [CLS] token's attention to every token. Keep only its
        #!MSG    attention to the 64 patches (drop column 0, the [CLS] token itself).
        h = model.patch_embed(x)
        cls = model.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1) + model.pos_embed
        for previous_layer in layers[:layer]:
            h = previous_layer(h)

        _, attn_weights = layers[layer].self_attn(
            h, h, h, need_weights=True, average_attn_weights=True,
        )
        cls_to_patches = attn_weights[:, 0, 1:]
        #!TAG HWEND
    return cls_to_patches

N_SAMPLES = 4
sample_images = torch.stack([test_dataset[i][0] for i in range(N_SAMPLES)])
sample_labels = [CIFAR_CLASSES[test_dataset[i][1]] for i in range(N_SAMPLES)]
n_layers = len(vit_model.encoder.layers)
cls_attention = torch.stack([get_cls_attention(vit_model, sample_images, l) for l in range(n_layers)])
print(cls_attention.shape)  # (layers, images, 64) -- one attention weight per patch
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Before looking at pictures, two numbers per layer. If the `[CLS]` token attended to all
64 patches equally, every weight would be $1/64$ and the entropy of each attention row
would be $\log 64 \approx 4.16$. We print how far the largest weight is above that
uniform value, and the average entropy:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
for l in range(n_layers):
    a = cls_attention[l]
    entropy = -(a / a.sum(1, keepdim=True) * (a / a.sum(1, keepdim=True)).log()).sum(1).mean()
    print('layer {}: largest weight = {:.1f} x uniform, entropy = {:.2f} (uniform: 4.16)'
          .format(l, float((a.max(1).values * 64).mean()), float(entropy)))
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def unnormalize(img: Tensor) -> Tensor:
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)

# One shared color scale for every map - otherwise each map would be stretched to its
# own min/max and even a perfectly flat map would look like it had strong structure.
vmax = float(cls_attention.max())
fig, axes = plt.subplots(N_SAMPLES, n_layers + 1, figsize=(2.4 * (n_layers + 1), 2.4 * N_SAMPLES))
for i in range(N_SAMPLES):
    axes[i, 0].imshow(unnormalize(sample_images[i]).permute(1, 2, 0))
    axes[i, 0].set_title(sample_labels[i])
    for l in range(n_layers):
        im = axes[i, l + 1].imshow(cls_attention[l, i].view(8, 8), cmap='viridis', vmin=0, vmax=vmax)
        axes[i, l + 1].set_title(f'layer {l}')
    for ax in axes[i]:
        ax.axis('off')
fig.colorbar(im, ax=axes, shrink=0.6, label='attention weight')
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The first layer should look almost perfectly flat - and almost the same for every
image. In the later layers, the `[CLS]` token concentrates on a few patches (when this
lab was tested, the largest weight was roughly 7-12 times the uniform value) and which
patches it picks differs from image to image. Whether those patches are actually "on the
object" is worth checking with your own eyes rather than assuming - with this little
training, often they won't be.

**Question:** Why is the first layer's `[CLS]` attention almost the same for every
image?
#!TAG HWBEGIN

**Answer:** In layer 0, the `[CLS]` token's input is `cls_token + pos_embed[0]` - the
same vector for every image, so its query is identical for every image too. Any
difference could only come from the patch keys, and the trained model barely uses them
here. That is consistent with the model collecting information into `[CLS]` only in
later layers, after the patches have exchanged information with each other and the
`[CLS]` token's own query depends on the image.
#!TAG HWEND

## Conclusion

Here's what you should take away from this lab:

 - Patch embedding (Exercise 1) is genuinely simple: a single strided convolution turns
an image into a sequence of tokens, and everything after that is the same Transformer
encoder Session 7 already derived - "a Transformer for images" really is that direct.
 - The CNN-vs-ViT gap under a small data/epoch budget (Exercise 2) is the measurable
version of the inductive-bias argument Session 10 made in words: convolutions build in
assumptions about images that a ViT has to learn from data instead. Here that shows up
twice - the ViT fits the training data more slowly, *and* it transfers less of what it
has learned to new images.
 - Attention weights (Exercise 3) are directly inspectable: you can ask "what is this
token looking at?" and get a literal answer. But read them carefully - use one shared
color scale, check them against the uniform baseline, and compare layers, before
concluding that a model "looks at" anything.
<!-- #endregion -->
