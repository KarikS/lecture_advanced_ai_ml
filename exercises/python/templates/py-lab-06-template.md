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

> Adapted from the Hugging Face LLM Course (`huggingface/course`, commit `8b893f0`, and its
> notebooks `huggingface/notebooks`, commit `3468b87`), Apache License 2.0: Chapter 7.3
> "Fine-tuning a masked language model" (Exercises 1-3) and Chapters 3.2 and 3.4
> "Processing the data" and "A full training loop" (Exercise 4). Code and prose are adapted
> and shortened. Changed from the source: a much smaller BERT (BERT-mini, 11M parameters,
> instead of DistilBERT/BERT-base) and smaller data subsets, so that every training run
> takes about 2-3 minutes on a CPU; the masking and the evaluation are written by hand
> instead of using library helpers; no Hugging Face Hub login or upload.

Session 8 showed how BERT is *pretrained* with masked language modeling (MLM) and then
*fine-tuned* for a downstream task through its `[CLS]` token. This lab does both with a real
pretrained BERT: you look at what it predicts for a `[MASK]`, build the MLM training data
yourself, continue its pretraining on movie reviews (*domain adaptation*), and fine-tune it
to detect paraphrases.

> **Instructor timing (2h session).** ~15 min: intro, setup and downloads (model ~45 MB,
> IMDB ~85 MB, MRPC ~1 MB). ~15 min: Exercise 1 - `[MASK]` predictions by hand. ~30 min:
> Exercise 2 - chunking and BERT's masking scheme. ~25 min: Exercise 3 - domain adaptation
> (a ~2.5 minute run). ~30 min: Exercise 4 - paraphrase detection (a ~2 minute run).
> ~5 min buffer.

## Setup

These labs are designed to run on [Google Colab](https://colab.research.google.com/) -
no local Python install needed. Open this notebook via Colab's GitHub loader
(`colab.research.google.com/github/KarikS/lecture_advanced_ai_ml/blob/main/<path-to-this-notebook>`)
or File > Open notebook > GitHub tab, repo `KarikS/lecture_advanced_ai_ml`, then run
the cell below once per session. Running the notebook locally instead (e.g. via the
`exercises/python/` venv described in that folder's README) works unchanged - the cell
below is a no-op there. No GPU is needed - each of the two training runs took 2-3 minutes
with 4 CPU threads in testing. Colab already has `transformers` and `datasets` installed.
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
import math
import time

import datasets
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorForLanguageModeling, DataCollatorWithPadding)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()  # hide the expected "unused weights" notes
transformers_logging.disable_progress_bar()
datasets.disable_progress_bars()
torch.manual_seed(0)
torch.set_num_threads(4)

CHECKPOINT = 'google/bert_uncased_L-4_H-256_A-4'
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1: What Does a Pretrained BERT Predict?

The source uses DistilBERT; we use **BERT-mini**, one of the small BERT models Google released
alongside the original BERT: the same architecture and pretraining (MLM and next sentence
prediction on English Wikipedia and BookCorpus), with 4 layers, $d_\text{model} = 256$ and 4
heads - 11M parameters instead of BERT-base's 110M. The checkpoint includes the pretrained
MLM head, so we can ask it to fill in a `[MASK]` right away.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
mlm_model = AutoModelForMaskedLM.from_pretrained(CHECKPOINT)
print(f'{mlm_model.num_parameters() / 1e6:.1f}M parameters')

inputs = tokenizer('This is a great [MASK].', return_tensors='pt')
print(inputs['input_ids'])
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The tokenizer has added `[CLS]` at the start and `[SEP]` at the end. The model returns one
vector of logits over the 30,522-token vocabulary for *every* position; to fill in the mask,
we need the logits at the `[MASK]` position.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def top_mask_predictions(model, text: str, k: int = 5) -> list[str]:
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab_size)
    #!TAG HWBEGIN
    #!MSG TODO: Find the position of the [MASK] token (tokenizer.mask_token_id) in
    #!MSG inputs['input_ids'], take the logits at that position, and return the k tokens
    #!MSG with the highest logits as strings (torch.topk, tokenizer.decode([token_id])).
    mask_position = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0][0]
    top_ids = torch.topk(logits[0, mask_position], k).indices.tolist()
    return [tokenizer.decode([token_id]) for token_id in top_ids]
    #!TAG HWEND


PROBE_SENTENCES = ['This is a great [MASK].',
                   'I went to the [MASK] to buy some bread.',
                   'The actor gave a [MASK] performance.']
for sentence in PROBE_SENTENCES:
    print(f'{sentence:45s} {top_mask_predictions(mlm_model, sentence)}')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
**Question:** As humans, we can think of many completions for "This is a great [MASK]." -
"day", "ride", "painting". What decides which ones the model predicts?
#!TAG HWBEGIN

**Answer:** The corpus it was pretrained on. MLM teaches the model the statistical patterns
of its training text, here English Wikipedia and BookCorpus, so its predictions are fairly
generic, everyday or encyclopedic words. A model that should work well on a particular kind
of text (legal contracts, scientific articles, movie reviews) may treat that domain's typical
words as rare - which is what domain adaptation in Exercise 3 addresses.
#!TAG HWEND

## Exercise 2: Preparing the Data for Masked Language Modeling

To adapt the model to movie reviews, we use the IMDb dataset (Large Movie Review Dataset), a
corpus of highly polarized reviews usually used for sentiment analysis. We don't need its
labels for language modeling. The source trains on 10,000 chunks; we take 2,000 reviews.
The training split is sorted by label, so we shuffle before selecting.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
imdb = load_dataset('stanfordnlp/imdb', split='train').shuffle(seed=42).select(range(2000))
print(imdb)
print(imdb[0]['text'][:300])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
For both autoregressive and masked language modeling, a common preprocessing step is to
**concatenate all examples and split the whole corpus into chunks of equal size**, instead of
tokenizing each example on its own: individual examples would otherwise be truncated if they
are too long, losing text that is useful for language modeling, or padded if they are short.
So we tokenize *without* truncation, then group the tokens into chunks of 128 (BERT's context
is 512 tokens, but shorter chunks train faster). The last, incomplete chunk is dropped. For
MLM, the labels start out as a copy of the inputs - the masking happens later.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
CHUNK_SIZE = 128

tokenized_imdb = imdb.map(lambda batch: tokenizer(batch['text']), batched=True,
                          remove_columns=['text', 'label'])
print(tokenized_imdb)
print('first review lengths:', [len(ids) for ids in tokenized_imdb[:5]['input_ids']])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    #!TAG HWBEGIN
    #!MSG TODO: `examples` maps each feature name (input_ids, token_type_ids, attention_mask)
    #!MSG to a list of tokenized reviews. For every feature:
    #!MSG 1. concatenate all reviews into one long list,
    #!MSG 2. cut it into chunks of CHUNK_SIZE, dropping the last incomplete chunk.
    #!MSG Return a dict with the chunked features plus a 'labels' entry that copies the
    #!MSG chunked input_ids.
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated['input_ids']) // CHUNK_SIZE * CHUNK_SIZE
    result = {k: [t[i:i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
              for k, t in concatenated.items()}
    result['labels'] = result['input_ids'].copy()
    return result
    #!TAG HWEND


lm_dataset = tokenized_imdb.map(group_texts, batched=True).train_test_split(test_size=0.1, seed=42)
lm_dataset.set_format('torch')
print(lm_dataset)
assert all(len(ids) == CHUNK_SIZE for ids in lm_dataset['train'][:100]['input_ids'])
print(tokenizer.decode(lm_dataset['train'][1]['input_ids']))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Grouping produced many more examples than the 2,000 reviews, and a chunk can span the end of
one review and the start of the next - look for the `[SEP]` and `[CLS]` tokens in the middle
of the decoded chunk above.

**BERT's masking scheme** (Session 8). The labels are still identical to the inputs - we are
missing the key step: inserting `[MASK]` tokens at random positions. BERT selects 15% of the
tokens as prediction targets, and of those replaces 80% by `[MASK]`, 10% by a random token,
and leaves 10% unchanged. Only the selected tokens count in the loss: all other labels are
set to `-100`, the index that PyTorch's cross-entropy (and so the model's loss) ignores.
Special tokens (`[CLS]`, `[SEP]`) are never selected.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def mask_tokens(input_ids: Tensor, mlm_probability: float = 0.15,
                generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """BERT-style masking for a batch of token IDs. Returns (masked inputs, labels)."""
    labels = input_ids.clone()
    inputs = input_ids.clone()
    special = torch.tensor([tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                            for ids in input_ids.tolist()], dtype=torch.bool)
    #!TAG HWBEGIN
    #!MSG TODO:
    #!MSG 1. Select each non-special token with probability mlm_probability
    #!MSG    (torch.bernoulli on a probability tensor, with generator=generator).
    #!MSG 2. Set the labels of all non-selected tokens to -100.
    #!MSG 3. Of the selected tokens, replace 80% by tokenizer.mask_token_id, then half of the
    #!MSG    remaining selected ones (10% overall) by a random token ID
    #!MSG    (torch.randint(len(tokenizer), ...)); the rest stay unchanged.
    probabilities = torch.full(labels.shape, mlm_probability)
    probabilities[special] = 0.0
    selected = torch.bernoulli(probabilities, generator=generator).bool()
    labels[~selected] = -100

    to_mask = torch.bernoulli(torch.full(labels.shape, 0.8), generator=generator).bool() & selected
    inputs[to_mask] = tokenizer.mask_token_id

    to_randomize = (torch.bernoulli(torch.full(labels.shape, 0.5), generator=generator).bool()
                    & selected & ~to_mask)
    random_ids = torch.randint(len(tokenizer), labels.shape, generator=generator)
    inputs[to_randomize] = random_ids[to_randomize]
    #!TAG HWEND
    return inputs, labels
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def masking_statistics(inputs: Tensor, labels: Tensor, original: Tensor) -> str:
    selected = labels != -100
    return ('selected {:.1%} of all tokens; of those: [MASK] {:.1%}, random {:.1%}, unchanged {:.1%}'
            .format(float(selected.float().mean()),
                    float((inputs[selected] == tokenizer.mask_token_id).float().mean()),
                    float(((inputs[selected] != tokenizer.mask_token_id)
                           & (inputs[selected] != original[selected])).float().mean()),
                    float((inputs[selected] == original[selected]).float().mean())))


sample_ids = lm_dataset['train'][:256]['input_ids']
masked, labels = mask_tokens(sample_ids, generator=torch.Generator().manual_seed(0))
print('ours:        ', masking_statistics(masked, labels, sample_ids))

# Hugging Face's own collator implements the same scheme - the source course uses it directly.
hf_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
hf_batch = hf_collator([{'input_ids': ids} for ids in sample_ids])
print('Hugging Face:', masking_statistics(hf_batch['input_ids'], hf_batch['labels'], sample_ids))

print(tokenizer.decode(masked[0][:60]))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Both should select about 15% of the tokens (slightly less, because special tokens are never
selected), with about 80% / 10% / 10% among those. A random token can by chance be the
original one, so "unchanged" comes out a little above 10%.

**Question:** Why doesn't BERT simply replace *all* selected tokens by `[MASK]`?
#!TAG HWBEGIN

**Answer:** The `[MASK]` token never occurs during fine-tuning (Session 8). If the model only
ever had to predict at `[MASK]` positions, it would learn useful representations mainly for
those positions, and fine-tuning inputs would look different from everything it saw during
pretraining. With 10% random and 10% unchanged tokens, the model cannot tell which input
tokens are targets, so it has to build a good representation of *every* token - including
ones that look perfectly normal. (The source course also shows *whole word masking*, which
masks all sub-word tokens of a word together.)
#!TAG HWEND

## Exercise 3: Domain Adaptation - Continuing BERT's Pretraining on Movie Reviews

Fine-tuning a pretrained language model on in-domain text before training a task-specific
head is called **domain adaptation**; it was popularized by ULMFiT in 2018. Its loss is the
MLM loss on the masked positions, and we report its exponential, the **perplexity**: a model
that is not "perplexed" by unseen text assigns high probability to the right tokens.

Since masking is random, the validation perplexity would change from one evaluation to the
next. The source course's solution: mask the validation set *once*, with a fixed seed, and
reuse it. We also use 256 validation chunks to keep evaluation fast.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
val_ids = lm_dataset['test'][:256]['input_ids']
val_inputs, val_labels = mask_tokens(val_ids, generator=torch.Generator().manual_seed(1234))
val_batches = [(val_inputs[i:i + 64], val_labels[i:i + 64]) for i in range(0, len(val_ids), 64)]


def mlm_perplexity(model) -> float:
    model.eval()
    with torch.no_grad():
        losses = [float(model(input_ids=x, labels=y).loss) for x, y in val_batches]
    model.train()
    return math.exp(sum(losses) / len(losses))


print(f'perplexity before domain adaptation: {mlm_perplexity(mlm_model):.2f}')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The training loop is the same as for any PyTorch model - Hugging Face models compute their
own loss when `labels` are passed. New masks are drawn for every batch, so the model never
sees the same masking of a chunk twice. We train for 60 steps of 32 chunks (about 1,900 of
the ~3,900 training chunks), with AdamW and a learning rate of $10^{-4}$ (the source: $5
\cdot 10^{-5}$ for a much longer run).
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
MLM_STEPS, MLM_BATCH, MLM_LR = 60, 32, 1e-4

mask_generator = torch.Generator().manual_seed(0)
train_loader = DataLoader(lm_dataset['train']['input_ids'], batch_size=MLM_BATCH, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=MLM_LR)

mlm_model.train()
t0 = time.time()
for step, batch_ids in enumerate(train_loader, start=1):
    #!TAG HWBEGIN
    #!MSG TODO: Mask the batch with your mask_tokens (generator=mask_generator), compute
    #!MSG the model's loss (model(input_ids=..., labels=...).loss), backward pass, optimizer
    #!MSG step, zero the gradients. Call the loss `loss`.
    inputs, labels = mask_tokens(batch_ids, generator=mask_generator)
    loss = mlm_model(input_ids=inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #!TAG HWEND
    if step % 10 == 0:
        print(f'STEP: {step:3d}\tTRAIN LOSS: {loss.item():.3f}\tELAPSED: {time.time() - t0:.0f}s')
    if step == MLM_STEPS:
        break

print(f'perplexity after domain adaptation: {mlm_perplexity(mlm_model):.2f}')
for sentence in PROBE_SENTENCES:
    print(f'{sentence:45s} {top_mask_predictions(mlm_model, sentence)}')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
For reference (seeds 0, 1, 2, one CPU run each): perplexity drops from 21.26 to 18.9-19.3 in
about 2.5 minutes of training. "This is a great [MASK]." changes from *success, deal, bird,
flood, legend* to answers such as *film, movie, story* - but also *mistake* and *disaster*:
the model has learned what IMDb reviews talk about, not whether they are positive.

**Question:** Lab 7 reports a perplexity for a GPT-style model too. Why can't you compare
the two numbers?
#!TAG HWBEGIN

**Answer:** They measure different tasks. The GPT-style perplexity is about predicting each
next token from its left context only. The MLM perplexity here is about predicting 15% of the
tokens from context on *both* sides - much more information per prediction - and it also
depends on the masking scheme (80/10/10) and on the tokenizer: Lab 7 predicts bytes, BERT
WordPiece tokens. A perplexity is only comparable between models evaluated on the same
targets, with the same tokenizer.
#!TAG HWEND

## Exercise 4: Fine-Tuning BERT for Paraphrase Detection

Now the second half of Session 8: fine-tuning. We use **MRPC** (Microsoft Research Paraphrase
Corpus, part of the GLUE benchmark): pairs of sentences with a label saying whether they mean
the same thing. It is small - 3,668 training pairs and 408 validation pairs - so it trains
quickly. We start again from the original pretrained checkpoint, not the domain-adapted one:
MRPC consists of news sentences, not movie reviews.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
mrpc = load_dataset('nyu-mll/glue', 'mrpc')
print(mrpc)
print(mrpc['train'].features['label'])
print(mrpc['train'][0])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can't pass two separate sentences to the model - they have to be handled as a pair. Given
two texts, the tokenizer builds the input BERT expects, `[CLS] sentence1 [SEP] sentence2
[SEP]`, and the `token_type_ids` that tell the model which part is the first sentence (type 0)
and which is the second (type 1) - BERT learned these segment embeddings during pretraining,
through next sentence prediction (Session 8).
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
pair = tokenizer(mrpc['train'][0]['sentence1'], mrpc['train'][0]['sentence2'])
for token, token_type in list(zip(tokenizer.convert_ids_to_tokens(pair['input_ids']), pair['token_type_ids']))[:40]:
    print(f'{token}:{token_type}', end='  ')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We tokenize every pair (no padding yet), remove the text columns and rename `label` to
`labels`, the argument name the model expects. Padding happens per batch: **dynamic padding**
pads each batch only to its own longest pair, instead of padding everything to the longest
pair in the dataset. `DataCollatorWithPadding` does this.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def tokenize_pairs(batch: dict) -> dict:
    #!TAG HWBEGIN
    #!MSG TODO: Tokenize batch['sentence1'] and batch['sentence2'] as pairs, with
    #!MSG truncation=True.
    return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True)
    #!TAG HWEND


mrpc_tokenized = (mrpc.map(tokenize_pairs, batched=True)
                  .remove_columns(['sentence1', 'sentence2', 'idx'])
                  .rename_column('label', 'labels'))
padding_collator = DataCollatorWithPadding(tokenizer)
cls_train_loader = DataLoader(mrpc_tokenized['train'], batch_size=16, shuffle=True,
                              collate_fn=padding_collator, generator=torch.Generator().manual_seed(0))
cls_val_loader = DataLoader(mrpc_tokenized['validation'], batch_size=64, collate_fn=padding_collator)

batch = next(iter(cls_train_loader))
print({k: tuple(v.shape) for k, v in batch.items()})
```

<!-- #region pycharm={"name": "#%% md\n"} -->
`AutoModelForSequenceClassification` puts a new, randomly initialized classification head on
top of BERT's `[CLS]` representation - the fine-tuning setup from Session 8. The two
possible errors of a paraphrase detector matter differently depending on the use, and the
classes are imbalanced, so GLUE reports **F1** for MRPC alongside accuracy:
$F_1 = \frac{2\,TP}{2\,TP + FP + FN}$, with "paraphrase" as the positive class.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def evaluate_classifier(model) -> tuple[float, float]:
    model.eval()
    n_correct = n_total = tp = fp = fn = 0
    with torch.no_grad():
        for batch in cls_val_loader:
            #!TAG HWBEGIN
            #!MSG TODO: Predict the class of every pair (argmax over model(**batch).logits)
            #!MSG and update n_correct, n_total, tp, fp, fn (positive class: label 1).
            predictions = model(**batch).logits.argmax(dim=-1)
            labels = batch['labels']
            n_correct += int((predictions == labels).sum())
            n_total += len(labels)
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
            #!TAG HWEND
    model.train()
    return n_correct / n_total, 2 * tp / (2 * tp + fp + fn)


majority = sum(mrpc['validation']['label']) / len(mrpc['validation'])
print(f'majority baseline (always "paraphrase"): accuracy {majority:.3f}, F1 {2 * majority / (1 + majority):.3f}')

torch.manual_seed(0)
cls_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
print('before fine-tuning: accuracy {:.3f}, F1 {:.3f}'.format(*evaluate_classifier(cls_model)))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The training loop follows the source: AdamW, and a learning rate that decays linearly from its
maximum to 0 over all training steps (the `Trainer`'s default). The source trains BERT-base
for 3 epochs at $5 \cdot 10^{-5}$; for our much smaller model, 2 epochs at $10^{-4}$ worked
better in testing.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
CLS_EPOCHS, CLS_LR = 2, 1e-4

optimizer = torch.optim.AdamW(cls_model.parameters(), lr=CLS_LR)
total_steps = CLS_EPOCHS * len(cls_train_loader)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / total_steps)

cls_model.train()
t0 = time.time()
for epoch in range(1, CLS_EPOCHS + 1):
    for batch in cls_train_loader:
        #!TAG HWBEGIN
        #!MSG TODO: One training step: forward pass (the model returns .loss when the batch
        #!MSG contains labels), backward pass, optimizer step, scheduler step, zero the
        #!MSG gradients.
        loss = cls_model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        #!TAG HWEND
    accuracy, f1 = evaluate_classifier(cls_model)
    print(f'EPOCH: {epoch}\tVAL ACCURACY: {accuracy:.3f}\tVAL F1: {f1:.3f}\tELAPSED: {time.time() - t0:.0f}s')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
For reference (seeds 0, 1, 2): before fine-tuning, accuracy is 0.42-0.60 and F1 is
0.41-0.73. After two epochs (about 2-2.5 minutes on CPU), accuracy is 0.77-0.79 and F1 is
0.84-0.85. Always answering "paraphrase" gets accuracy 0.684 and F1 0.812.

**Question:** Before fine-tuning, the classifier is *worse* than always answering
"paraphrase". Why - BERT is pretrained, after all?
#!TAG HWBEGIN

**Answer:** Only the BERT body is pretrained. The classification head on top of the `[CLS]`
representation is new and randomly initialized, so its predictions mean nothing until it is
trained. Fine-tuning trains the head and, at the same time, adapts all of BERT's weights to
the task - which is why a few minutes on a small dataset are enough, while training the same
architecture from scratch on 3,668 pairs would get nowhere near this.
#!TAG HWEND

**Question:** Why report F1 next to accuracy for MRPC? Look at the majority baseline.
#!TAG HWBEGIN

**Answer:** About two thirds of the pairs are paraphrases, so a model that always says
"paraphrase" already gets about 68% accuracy - and an F1 of about 0.81, because it never
misses a paraphrase. Neither number alone tells you whether the model has learned anything;
both have to be compared against that baseline. F1 also ignores true negatives, so it focuses
on how well the positive class is found and how many false alarms come with it.
#!TAG HWEND

## Conclusion

Here's what you should take away from this lab:

 - A pretrained masked language model fills in a `[MASK]` from the logits at that position
(Exercise 1), and its predictions reflect the corpus it was pretrained on.
 - BERT's training data is simple to build: concatenate the corpus, cut it into chunks, and
select 15% of the tokens with the 80/10/10 replacement scheme (Exercise 2).
 - Continuing the MLM training on in-domain text (domain adaptation) shifts the model's
predictions towards that domain within a few minutes (Exercise 3).
 - Fine-tuning puts a new head on the `[CLS]` token and trains everything for the task;
judge the result against a baseline, with metrics that suit the class balance (Exercise 4).
<!-- #endregion -->
