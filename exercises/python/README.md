# Advanced AI & ML - Python Labs

This directory contains the PyTorch lab sessions for this course.

Lab 1 (`templates/py-lab-01-template.md`) is adapted from `py-lab-06-template.md`
("Lab 5") of the `lecture_i2dl` Python labs, cut down from a ~4h lab to a 2h session
and with the IMDB loader rewritten to avoid a Keras/TensorFlow dependency (see
"Dataset" below). Exercises 3-4 of the source lab (quadratic error surfaces,
gradient descent with momentum) were dropped entirely - that material belongs with
this course's own optimizers session instead.

Lab 2 (`templates/py-lab-02-template.md`) is an optional follow-up to Lab 1: Exercise 1
is adapted from `py-lab-02-template.md` ("Lab 1") and Exercise 2 from
`py-lab-04-template.md` ("Lab 3") of the `lecture_i2dl` Python labs. Both source labs
also include pen-and-paper derivation exercises (a separability proof for Exercise 1,
the general backpropagation derivation for Exercise 2) - those are dropped here in
favor of a fully hands-on, code-only version: you're given the results, your job is to
turn them into working code.

Lab 4 (`templates/py-lab-04-template.md`, Session 6) is a recap lab spanning RNN, VAE,
and GAN, one exercise per model family. `lecture_i2dl`'s own Python labs were checked
first, but none fit directly:

- Exercise 1 (vanilla RNN vs. LSTM on the "adding problem", a synthetic long-range
  dependency task from Hochreiter & Schmidhuber, 1997) is a fresh implementation.
  `py-lab-08-template.md` ("Lab 7") is a pen-and-paper BPTT derivation, too deep for a
  recap lab. Extending `py-lab-09-template.md` ("Lab 8")'s LSTM-on-IMDB exercise with a
  vanilla-RNN baseline was tried, but neither model beat chance accuracy within a
  lab-feasible budget (a 5,000-review subset, 3 epochs), so the comparison showed
  nothing.
- Exercise 2 (a 2D-latent VAE on MNIST) is a fresh implementation; its bonus cell
  (deliberately overweighting the KL term to induce posterior collapse) is adapted from
  `py-lab-11-template.md` ("Lab 10")'s VAE exercise.
- Exercise 3 (a GAN on a toy ring of 2D Gaussians, chosen so the generated distribution
  can be plotted directly at every stage of training) is a fresh implementation. A
  real-image GAN was avoided, partly because `py-lab-12-template.md` ("Lab 11 - currently
  not in use")'s own MNIST GAN exercise has been retired in the source course.

Lab 5 (`templates/py-lab-05-template.md`, Session 7) is not adapted from either source
course's Python labs. The toy task and its qualitative result (a pure `Conv1d` stack
fails at a task requiring long-range routing; a single self-attention layer solves it;
the same attention layer then fails a harder position-dependent version of the task
until a positional encoding is added) are adapted from François Fleuret's *Deep
Learning* course (University of Geneva, `fleuret.org/dlc`, §13.2 "Attention
Mechanisms") - licensed CC BY-NC-SA 4.0, so this lab specifically (unlike Labs 1-4)
carries that non-commercial/share-alike license rather than this repository's usual
terms. The dataset generator, model code, and training loop are a fresh implementation
for this course; Fleuret's own code for this task was not published alongside his
slides.

Lab 6 (`templates/py-lab-06-template.md`, Session 8) pretrains and fine-tunes BERT: `[MASK]`
predictions by hand, BERT's 15% / 80-10-10 masking scheme, domain adaptation by continued
MLM training on IMDb movie reviews, and fine-tuning for paraphrase detection on GLUE/MRPC.
Adapted from the Hugging Face LLM Course (`huggingface/course`, commit `8b893f0`, with its
notebooks in `huggingface/notebooks`, commit `3468b87`), Chapter 7.3 "Fine-tuning a masked
language model" and Chapters 3.2 and 3.4, licensed Apache 2.0 (verified from both repos'
`LICENSE` files); code and prose are modified and shortened. Changed from the source: the
small BERT-mini checkpoint (`google/bert_uncased_L-4_H-256_A-4`) instead of
DistilBERT/BERT-base, and smaller data subsets, so that each training run takes 2-3 minutes
on CPU; masking and evaluation written by hand instead of library helpers; no Hub login or
upload. This is the only lab that needs `transformers` and `datasets`.

Lab 7 (`templates/py-lab-07-template.md`, Session 9) builds the "simple, modern variant"
Transformer LM from Session 9 (pre-norm RMSNorm, SwiGLU, RoPE, no biases) component by
component, then trains it, samples from it and removes RoPE again as an ablation. Adapted
from Stanford CS336 *Language Modeling from Scratch*, Spring 2025, Assignment 1
(`stanford-cs336/assignment1-basics`, tag `spring2025`, commit `430e2c8`), Copyright 2025
Stanford University, licensed MIT (verified from the `LICENSE` file at that tag). Exercises
1-4 follow the handout's §3 problem by problem; Exercises 5-6 follow §4.1, §5.1, §6 and §7.3
(the NoPE ablation). Every model component is checked against the assignment's own snapshot
tests. The snapshots, the reference weights they were computed with, and the 5 MB TinyStories
sample used for training are downloaded at runtime from that pinned commit into `.data/cs336/`
rather than committed here. Deviations from the source, all for a CPU-feasible 2h session: no
BPE tokenizer (the handout's §2; the lab uses raw UTF-8 bytes as tokens), PyTorch's AdamW
instead of a hand-written one (§4.3), and a much smaller model and training budget; the exact
numbers are in the template's header note and Exercise 5.

Lab 8 (`templates/py-lab-08-template.md`, Session 10) trains a small Vision Transformer
from scratch on a CIFAR-10 subset, alongside a CNN baseline trained under the same
data/epoch budget, so the CNN-vs-ViT inductive-bias/data-hunger argument from Session
10's slides is reproduced on real data rather than only argued. Adapted from the UvA
Deep Learning Tutorials' "Vision Transformers" notebook (`phlippe/uvadlc_notebooks`,
Tutorial 15), licensed MIT (verified directly from the repo's own `LICENSE.md`) - scaled
down substantially from the source tutorial's 180-epoch/full-CIFAR-10 run (under an hour
on an NVIDIA TitanRTX) to a CPU-feasible subset/epoch budget for a 2h session; the exact
numbers are documented in the template's own header note.

## USER: How to get started

Every lab session is contained in one notebook. The recommended way to run them is
Google Colab - no local Python install needed:

1. Open the lab's question notebook via Colab's GitHub loader, e.g. for Lab 1:
   `https://colab.research.google.com/github/KarikS/lecture_advanced_ai_ml/blob/main/exercises/python/questions/py-lab-01-question.ipynb`
   (or in Colab: File > Open notebook > GitHub tab > `KarikS/lecture_advanced_ai_ml`)
2. Run the first "Setup" cell - it clones this repository so the notebook can reach
   `data/` and the rest of the course code. This step is a no-op if you're instead
   running the notebook locally (see below).
3. File > Save a copy in Drive early, so your progress survives a runtime disconnect.
4. Run the rest of the notebook top to bottom.

Each session gets its own notebook and link once it's built (`py-lab-02-question.ipynb`,
`py-lab-03-question.ipynb`, ...). GPU is optional for most
labs (the notebook will tell you when one is actually needed); if you want one anyway,
use Runtime > Change runtime type > GPU.

### Running locally instead

The same notebook also runs unchanged on your own machine - the Colab setup cell is a
no-op outside Colab. Make sure to execute all the following commands while inside
`exercises/python/`.

```shell
cd exercises/python/
python -m venv .venv
```

The freshly created environment needs to be activated.

```shell
source .venv/bin/activate
```

Now you can install the requirement dependencies over the `requirements.txt` file or
conveniently via the Makefile.

```shell
make requirements
```

To view the files in jupyter, start a notebook server in your environment.

```shell
jupyter notebook .
```

## Dataset

Lab 1 uses the IMDB sentiment dataset. Rather than pulling it in via
`keras.datasets.imdb` (which drags in a full TensorFlow/Keras install just to fetch
one dataset), `data/imdb.npz` and `data/imdb_word_index.json` are committed directly
in this repo - the exact same files Keras itself downloads and caches, loaded here
with a few lines of numpy instead. No TensorFlow/Keras dependency anywhere in this
directory.

## DEVELOPER: How to get started

For better version control that notebooks are kept in markdown format.
Apart from the `requirements.txt` you will need to install the `jupytext` package over
pip.

You can directly edit the template `.md` files or convert them to `.ipynb` files with
`make convert-md-to-ipynb FILE=<file>`. Conversion back to `.md` is done via
`make convert-ipynb-to-md FILE=<file>`.

The python script `convert.py` can be utilized to produce all files for questions
and solutions (including `.md`, `.ipynb` and `.pdf`):

```shell
python convert.py -f <file>
```
If no `-f` flag is provided, all template markdown files will be converted.

PDF generation needs a LaTeX install (`nbconvert` + `pdflatex`) and fills in a
title/author from the `COURSE_TITLE` / `COURSE_TERM` / `LECTURERS` constants at the
top of `convert.py` - set those to your own course run before generating PDFs; they're
left blank by default rather than guessed.

Some tags in the template lead to different behavior in question and solution
versions:

- `#!TAG HWBEGIN` and `#!TAG HWEND`: Everything in between does two tags is treated as
homework for the student. The content in this block is only included in the solutions.
- `#!MSG <string>`:  This allows to include messages within the homework block and could
contain some hints for the student or assignments like "fill this function". The message
is only contained in the questions.
- `#!TAG SKIPQUESTEXEC`: Include this at the top of a cell, which should not be executed
in the questions and only in the solutions. This becomes handy, when there are gaps for
homework or yet undefined variables/functions that would throw an error.
