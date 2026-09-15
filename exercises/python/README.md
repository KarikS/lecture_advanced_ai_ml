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
