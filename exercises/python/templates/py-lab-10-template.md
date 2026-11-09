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
# Lab 10

> The recipe for `sentence-transformers/all-MiniLM-L6-v2` follows that model's card on the
> Hugging Face Hub (Apache 2.0).

Training a model is only part of the work. You also want to *look* at what a model has
learned, and to keep track of which run used which settings and gave which result -
otherwise you cannot reproduce or compare your experiments a week later. This lab walks
through three widely used tools for that, on one small task:

1. embed news headlines with a small pretrained Transformer (MiniLM),
2. explore the embeddings visually in **TensorBoard**'s embedding projector,
3. train classifiers on the embeddings and track every run with **MLflow**,
4. (optional) log the same runs to **Weights & Biases** (W&B).

TensorBoard and MLflow run entirely on your machine (or Colab runtime): no account,
nothing leaves the computer. W&B is a hosted service; the optional exercise runs it in
offline mode by default.

## Working Together: Three Groups, Three Tools

For our group of seven, form groups of **2 + 2 + 3**. Everyone completes Exercise 1
individually; then each group becomes the guide for one tool. Exercises 2-4 provide
the preparation material for your group. W&B is optional for individual study, but
is Group 3's focus in this format.

| Group | Members | Tool and preparation | Show the class |
| --- | --- | --- | --- |
| 1 | 2 | **TensorBoard Projector**, Exercise 2 | Inspect labelled articles and neighbours; compare PCA and t-SNE; explain a misleading projection. |
| 2 | 2 | **MLflow**, Exercise 3 | Compare configurations across seeds, inspect metric histories and save reproducibility artifacts. |
| 3 | 3 | **Weights & Biases**, Exercises 3-4 | Build a comparison dashboard, inspect prediction errors and share a short results report. |

**Group 3 has three people** because it combines instrumenting the training loop,
organising the hosted dashboard and analysing example-level predictions. Divide these
three responsibilities among the members, then prepare the demonstration together.
Group 2 should use a local installation for MLflow's web UI; in Colab, use the Python
run table and plots instead. Group 3 can reuse Exercise 3's classifier helper and
training loop. Complete W&B signup before class.

**Each group gives a 10-15 minute live demonstration, with no question period or
audience tasks.** Everyone in the group presents part of it. No slide deck is needed.
Use roughly 2-3 minutes for the workflow and logging code, 6-9 minutes to demonstrate
results in the tool, and 2-3 minutes for your finding, limitation and recommendation.
Prepare runs and visualisations beforehand so the demonstration does not wait on training.

Show at least one finding supported by your own results, one limitation of the tool
or analysis, and a concrete situation in which you would use it. Share your notebook
and screenshots afterwards; Group 3 also shares an accessible project/report link.
Keep API keys out of shared material.

> **Instructor timing (2h session).** ~20 min: setup and individual work on Exercise 1.
> ~50 min: group preparation. ~30-45 min: three demonstrations of 10-15 minutes each,
> without Q&A. Use the remaining 5-20 minutes for transitions, finishing and sharing
> materials. Reuse the embeddings and existing small training budgets; do not increase
> dataset size or epochs for the group tasks. Aim for less than 10 minutes per training cell.

## Setup

These labs are designed to run on [Google Colab](https://colab.research.google.com/) -
no local Python install needed. Open this notebook via Colab's GitHub loader
(`colab.research.google.com/github/KarikS/lecture_advanced_ai_ml/blob/main/<path-to-this-notebook>`)
or File > Open notebook > GitHub tab, repo `KarikS/lecture_advanced_ai_ml`, then run
the cell below once per session. Running the notebook locally instead (e.g. via the
`exercises/python/` venv described in that folder's README) works unchanged - the cell
below is a no-op there. No GPU is needed.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    import os
    if not os.path.exists('lecture_advanced_ai_ml'):
        !git clone https://github.com/KarikS/lecture_advanced_ai_ml.git
    !pip install -q mlflow
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

set_matplotlib_formats('png', 'pdf')
torch.manual_seed(0)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1: Sentence Embeddings with MiniLM

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) contains news articles
(title and first sentence) in four categories: World, Sports, Business, Sci/Tech. We use
a small random subset: 1,000 articles for training and 500 for testing.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
ag_news = load_dataset('fancyzhx/ag_news')
label_names = ag_news['train'].features['label'].names
train_data = ag_news['train'].shuffle(seed=0).select(range(1000))
test_data = ag_news['test'].shuffle(seed=0).select(range(500))
print(label_names)
for example in train_data.select(range(3)):
    print(f"[{label_names[example['label']]}] {example['text'][:100]}...")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is a
6-layer Transformer encoder (22M parameters), trained so that texts with similar meaning
get similar vectors. Like BERT (Session 8), it outputs one 384-dimensional vector per
token. To get **one** vector per text, the model's card prescribes **mean pooling**:
average the token vectors, but only over real tokens - padding tokens (attention mask
0) must not count. Then normalise the result to unit length, so that the dot product of
two embeddings is their cosine similarity.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name).eval()


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """token_embeddings: (batch, seq_len, dim), attention_mask: (batch, seq_len) of 0/1."""
    #!TAG HWBEGIN
    #!MSG TODO: Multiply the token embeddings by the mask (broadcast it to the last
    #!MSG dimension), sum over the sequence dimension, and divide by the number of real
    #!MSG tokens per text (clamp it to at least 1e-9 to avoid dividing by 0).
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    #!TAG HWEND


@torch.no_grad()
def embed(texts: list[str], batch_size: int = 64) -> Tensor:
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=128, return_tensors='pt')
        token_embeddings = encoder(**batch).last_hidden_state
        sentence_embeddings = mean_pooling(token_embeddings, batch['attention_mask'])
        chunks.append(F.normalize(sentence_embeddings, p=2, dim=1))
    return torch.cat(chunks)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
# Sanity check: padding must not change a text's embedding.
alone = embed(['The match ended in a draw.'])
padded = embed(['The match ended in a draw.', 'A much longer second sentence forces padding of the first one.'])[:1]
print(torch.allclose(alone, padded, atol=1e-5))
assert torch.allclose(alone, padded, atol=1e-5)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
sentences = ['The striker scored twice in the final.',
             'The team won the championship game.',
             'The central bank raised interest rates.',
             'Stock markets fell after the announcement.']
e = embed(sentences)
print((e @ e.T).numpy().round(2))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The two sports sentences should be more similar to each other than to the two business
sentences, even though they share hardly any words. Now embed the whole subset.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
start = time.time()
X_train = embed(train_data['text'])
X_test = embed(test_data['text'])
y_train = torch.tensor(train_data['label'])
y_test = torch.tensor(test_data['label'])
print(X_train.shape, X_test.shape, f'{time.time() - start:.0f}s')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
**Question:** Why would plain averaging over *all* positions, padding included, be a bug -
and why would you not notice it on a single sentence?
#!TAG HWBEGIN

**Answer:** In a batch, shorter texts are padded to the length of the longest one. With
plain averaging, the padding positions' vectors (which the model still computes) are
mixed into the embedding, and the *same* text gets a different embedding depending on
what else is in its batch - the sanity check above would fail. A single sentence has no
padding, so averaging over all positions gives the right answer there; the bug only shows
up in batches.
#!TAG HWEND

## Exercise 2: Looking at the Embeddings in TensorBoard

### Group 1: Inspect Examples, Not Just Clusters

Extend the basic projector export below for your demonstration:

1. Export metadata with two columns, `category` and `text`, using
   `metadata_header=['category', 'text']`. Build one metadata row per test article;
   replace tabs and newlines in its text with spaces. Use category for colour and text
   to identify individual points. Keep metadata in the same order as `X_test`.
2. Pick one Business article and one Sci/Tech article. Compute their five nearest
   neighbours in the original embedding space with `X_test @ X_test[index]`, excluding
   the selected article itself. Show the texts, categories and cosine similarities.
3. Locate these examples in PCA and t-SNE. Compare the visible neighbours with the
   original-space neighbours; record the t-SNE settings used. Show one case where the
   projection makes the relationship look different. Projection distance is not cosine
   similarity in the original 384-dimensional space.

**Demo deliverable:** a labelled projector view, a small neighbour table for the two
selected articles and a concrete explanation of one overlap or projection distortion.
No additional model training is needed.


TensorBoard's **projector** shows high-dimensional vectors in 2D or 3D, using PCA, t-SNE
or UMAP, and can colour every point by a label. `SummaryWriter.add_embedding` writes the
vectors and one metadata label per vector into a log directory.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
log_dir = 'runs/lab10'
writer = SummaryWriter(log_dir)
#!TAG HWBEGIN
#!MSG TODO: Write the test embeddings with writer.add_embedding, passing the category
#!MSG name of every test example as metadata and tag='ag_news_minilm'.
writer.add_embedding(X_test, metadata=[label_names[i] for i in y_test],
                     tag='ag_news_minilm')
#!TAG HWEND
writer.close()
print(sorted(os.listdir(log_dir)))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now start TensorBoard and open the **Projector** tab (on the top right, it may be hidden
under "Inactive" in the drop-down).

- **Colab:** run the next cell; TensorBoard opens inside the notebook.
- **Locally:** in a terminal, inside `exercises/python/`, run
  `tensorboard --logdir runs` and open `http://localhost:6006`.

In the projector, choose "Color by: label". Try PCA first, then t-SNE (let it run for a
few hundred iterations).

> If the Projector tab stays empty in Colab, download the `runs/` folder (Files panel on
> the left) and open it with a local TensorBoard instead.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
if IN_COLAB:
    %load_ext tensorboard
    %tensorboard --logdir runs
```

<!-- #region pycharm={"name": "#%% md\n"} -->
**Question:** Do the four categories form separate clusters? Which two overlap most, and
does that make sense?
#!TAG HWBEGIN

**Answer:** Sports usually forms the most clearly separated cluster, and World is fairly
distinct too. Business and Sci/Tech overlap most: many articles are about technology
companies, products and markets, and could plausibly carry either label. The projection
is only a 2D/3D view of a 384-dimensional space, so overlap in the plot does not prove
the classes are inseparable - Exercise 3 measures that directly.
#!TAG HWEND

## Exercise 3: Tracking Experiments with MLflow

### Group 2: Compare Repeated Runs and Record Their Provenance

Extend the basic tracked experiment below:

1. Run the four `(lr, hidden)` settings with seeds 0, 1 and 2: 12 runs, still with
   20 epochs and the existing embeddings. Include the seed in each run name. Log
   the encoder name, dataset name, subset shuffle seed and relevant library versions
   alongside the existing parameters.
2. Query the runs and group by `(lr, hidden)`. Show the mean and standard deviation
   of final test accuracy across the three seeds, plus the individual results.
   Restrict the query to this experiment batch, for example with a shared batch tag,
   so older notebook runs do not silently enter the comparison.
3. Save a JSON file describing the model/data settings and library versions, and a
   CSV of the aggregate comparison. Log both with `mlflow.log_artifact` in a separate
   summary run. Show how to locate and download those artifacts.
4. Demonstrate parameter filtering and metric histories in the UI or Python. Explain
   why the highest single accuracy is weaker evidence than a consistent improvement
   across seeds, and label this comparison exploratory because it uses the test set.

**Demo deliverable:** a reproducibility artifact, a comparison across seeds and one
metric-history view that supports your conclusion. Reuse the embeddings for every run.


We now train classifiers on the frozen embeddings - a linear layer, and a small MLP - with
a few learning rates. That is already 4 runs, each with its own settings, curves and
final result. MLflow records all of this in a local database:

- `mlflow.set_tracking_uri('sqlite:///mlflow.db')` stores everything in a SQLite file in
  the current folder.
- `mlflow.set_experiment(name)` groups runs.
- Inside `with mlflow.start_run(run_name=...):`, `mlflow.log_params(dict)` records the
  settings, and `mlflow.log_metric(name, value, step=...)` records a value per epoch.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('lab10-agnews')


def make_classifier(hidden: int) -> nn.Module:
    if hidden == 0:
        return nn.Linear(384, 4)
    return nn.Sequential(nn.Linear(384, hidden), nn.ReLU(), nn.Linear(hidden, 4))


def train_tracked(lr: float, hidden: int, epochs: int = 20, batch_size: int = 64,
                  seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = make_classifier(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    with mlflow.start_run(run_name=f'lr={lr} hidden={hidden}'):
        #!TAG HWBEGIN
        #!MSG TODO: Log the hyperparameters (lr, hidden, epochs, batch_size, seed) and the
        #!MSG training set size as params.
        mlflow.log_params({'lr': lr, 'hidden': hidden, 'epochs': epochs,
                           'batch_size': batch_size, 'seed': seed, 'n_train': len(y_train)})
        #!TAG HWEND
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(len(y_train))
            for i in range(0, len(y_train), batch_size):
                idx = permutation[i:i + batch_size]
                loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss = F.cross_entropy(model(X_train), y_train).item()
                test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
            #!TAG HWBEGIN
            #!MSG TODO: Log train_loss and test_acc for this epoch (step=epoch).
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('test_acc', test_acc, step=epoch)
            #!TAG HWEND
        mlflow.log_metric('final_test_acc', test_acc)
    return test_acc
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
for lr in (1e-3, 1e-2):
    for hidden in (0, 128):
        acc = train_tracked(lr, hidden)
        print(f'lr={lr:<6} hidden={hidden:<4} final test accuracy={acc:.3f}')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
All four runs are now in `mlflow.db`. You can query them from Python - this works the
same in Colab and locally:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
runs = mlflow.search_runs(experiment_names=['lab10-agnews'], order_by=['metrics.final_test_acc DESC'])
runs[['tags.mlflow.runName', 'params.lr', 'params.hidden', 'metrics.final_test_acc']]
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
client = mlflow.tracking.MlflowClient()
plt.figure(figsize=(7, 4))
for _, run in runs.iterrows():
    #!TAG HWBEGIN
    #!MSG TODO: Fetch the test_acc history of this run with client.get_metric_history and
    #!MSG plot it against the step, labelled with the run name.
    history = client.get_metric_history(run['run_id'], 'test_acc')
    plt.plot([m.step for m in history], [m.value for m in history], label=run['tags.mlflow.runName'])
    #!TAG HWEND
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.legend()
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
**Locally**, MLflow also has a web UI: in a terminal, inside `exercises/python/`, run
`mlflow ui --backend-store-uri sqlite:///mlflow.db` and open `http://localhost:5000`.
Select the four runs and click "Compare" to see their parameters side by side and their
curves in one plot. (In Colab the UI is not reachable without extra tunnelling, which is
why the cells above query the runs from Python.)

**Question:** These runs log the hyperparameters. What else would you need to log to
reproduce the best run exactly, a month from now?
#!TAG HWBEGIN

**Answer:** Everything that influences the result: the random seed (logged above), the
exact data (dataset name and version, the subset and its shuffling seed), the embedding
model and its revision, the code version (e.g. the git commit, MLflow can record it
automatically when run from a script in a repository), the library versions, and ideally
the trained model itself as an artifact (`mlflow.log_artifact` or
`mlflow.pytorch.log_model`). Note also that "test accuracy" was measured after every epoch
here; picking the best run by it means the test set has been used for model selection.
#!TAG HWEND

**Question:** The best runs differ by about one or two percentage points. Is that a real
difference?
#!TAG HWBEGIN

**Answer:** Not necessarily. With 500 test examples, one percentage point is 5
articles, and the standard error of an accuracy around 0.87 is about
$\sqrt{0.87 \cdot 0.13 / 500} \approx 0.015$. Different seeds also give different
results. To decide, repeat each setting with several seeds (each a tracked run, so
MLflow can aggregate them) and compare means and spreads.
#!TAG HWEND

## Exercise 4 (optional): The Same Runs in Weights & Biases

### Group 3: Build a Shared Experiment Story

After signup and the basic logging exercise below, divide the extension among your
three members, then combine it into one demonstration:

1. **Logging:** run the four settings with seeds 0, 1 and 2 (12 small runs). Add the
   seed to run names and use `group` in `wandb.init` to collect seeds for the same
   configuration. Log the encoder/data settings as configuration. Keep epochs and
   dataset size unchanged and reuse the embeddings.
2. **Dashboard:** compare grouped accuracy curves with variation across seeds, and
   show how filtering changes the runs included. Prepare a compact comparison of
   final accuracy by configuration and seed. Explain both the typical result and
   the variability rather than selecting only the highest run.
3. **Error analysis and sharing:** inside one run, after training and before
   `run.finish()`, use the still-available classifier to predict `X_test`. Log a
   `wandb.Table` with up to 20 misclassified examples: article text, true category,
   predicted category and predicted-class softmax probability. Log a confusion-matrix
   plot for all 500 test examples. Find one recurring error and relate it to the
   embedding overlap. Treat softmax probability as a score, not calibrated certainty.

Create a short W&B report containing a comparison plot, the error table and a few
sentences stating your finding and its limitation. Check access to the project and
report with another group member's account before presenting. If online access fails,
prepare the same table and confusion matrix in the notebook and show saved screenshots.

**Demo deliverable:** a dashboard comparing repeated runs, an example-level error
analysis and a report link accessible to the class. All three members demonstrate
their contribution; finish with when you would choose W&B over local MLflow.


W&B does the same job as MLflow, but as a hosted service: runs are uploaded to
`wandb.ai`, where you get dashboards and can share them with a team. Students and
academics can apply for a free academic license (sign up with your university email).

### Before Class: Sign Up for the Online Demonstration

W&B offers free academic access for students and educators, including coursework.
Use the [academic signup/application route](https://site.wandb.ai/research/) with your
university email and follow the eligibility steps shown there. Check the current
[plans](https://site.wandb.ai/pricing/) if creating a shared workspace: the ordinary
free plan lists up to five model seats, so use the academic route for a workspace
that includes all seven students. Each student who logs in uses their own account;
do not share passwords or API keys.

1. Create your account before class and confirm that you can open your workspace.
2. Run the next cell once to install/import W&B. It defaults to **offline** mode,
   which writes to a local `wandb/` folder and needs no account.
3. For the online demonstration, change the `WANDB_MODE` assignment in that cell to
   `'online'` and run the cell again **before** starting training.
4. In a separate notebook code cell, run `wandb.login()`, then follow the login prompt.
   If asked for an API key, enter it through the hidden prompt; do not paste it into
   notebook source, slides or chat. Locally, `wandb login` in a terminal also works.
5. Run the training cell below. Open the project in your W&B workspace, compare the
   four runs, and check that the people you want to share with can access the link.
   If using a shared academic workspace, pass its name as `entity='your-team-name'`
   to `wandb.init(...)`; otherwise use your own workspace.

The online exercise uploads run metrics and configuration to W&B. If signup or
workspace access is unavailable, keep offline mode, demonstrate the logging code
and local run files, and use the MLflow plots for the results discussion. The hosted
W&B dashboard requires online runs and an account. You can upload offline runs later
with `wandb sync wandb/offline-run-*` after logging in.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
try:
    import wandb
except ImportError:
    %pip install -q wandb
    import wandb

os.environ['WANDB_MODE'] = 'offline'
# Keep W&B's end-of-run summary out of the notebook output.
os.environ['WANDB_SILENT'] = 'true'
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def train_wandb(lr: float, hidden: int, epochs: int = 20, batch_size: int = 64,
                seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = make_classifier(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    config = {'lr': lr, 'hidden': hidden, 'epochs': epochs, 'batch_size': batch_size,
              'seed': seed, 'n_train': len(y_train)}
    #!TAG HWBEGIN
    #!MSG TODO: Start a run with wandb.init(project='lab10-agnews', config=config,
    #!MSG name=...), and replace the MLflow logging by run.log({...}, step=epoch).
    #!MSG Finish the run at the end.
    run = wandb.init(project='lab10-agnews', config=config, name=f'lr={lr} hidden={hidden}')
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(y_train))
        for i in range(0, len(y_train), batch_size):
            idx = permutation[i:i + batch_size]
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = F.cross_entropy(model(X_train), y_train).item()
            test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
        run.log({'train_loss': train_loss, 'test_acc': test_acc}, step=epoch)
    run.summary['final_test_acc'] = test_acc
    run.finish()
    #!TAG HWEND
    return test_acc


for lr in (1e-3, 1e-2):
    for hidden in (0, 128):
        train_wandb(lr, hidden)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
**Question:** TensorBoard, MLflow, W&B - when would you pick which?
#!TAG HWBEGIN

**Answer:** There is no single right answer; typical reasons are:
TensorBoard for quick, local visualisation of one training run (curves, histograms, the
projector), built into PyTorch and TensorFlow. MLflow when you need a local or
self-hosted record of many runs with parameters, metrics and artifacts, plus a model
registry - nothing has to leave your infrastructure, which matters for confidential data
or company policy. W&B when a team wants a shared, hosted dashboard with little setup -
at the price of sending run data to an external (US-based) service and depending on its
pricing and terms. Many projects combine them.
#!TAG HWEND

## Conclusion

Here's what you should take away from this lab:

 - A pretrained sentence encoder plus mean pooling (Exercise 1) turns texts into vectors
in a few lines; the masking detail matters as soon as you embed batches.
 - Looking at representations (Exercise 2) is cheap with TensorBoard's projector and
often tells you quickly whether a representation separates what you care about.
 - Experiment tracking (Exercise 3) is a habit, not a library: log every setting that
can change the result, log curves rather than only the final number, and compare runs
from the record instead of from memory. MLflow does this locally with a single SQLite
file; W&B (Exercise 4) does the same as a hosted service.
<!-- #endregion -->
