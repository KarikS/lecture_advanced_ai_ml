# Advanced Artificial Intelligence and Machine Learning

This course is oriented towards students, professionals and anyone interested that aims to
deepen their understanding of what **mechanisms underlie modern deep learning architectures**,
especially Transformers. As very specific architectures are overtaken quickly, we try to introduce
principles shared by many such models wherever possible (e.g. dedicating a lot to optimization).

Compared to other courses in this area, which usually focus on classical architectures (CNNs, RNNs, AEs)
*or* Transformers or are guided by specific domains like Computer Vision or Natural Language Processing,
we specifically try to make the **balancing act of a domain-agnostic introduction of classical architectures
while giving an in-depth view on Transformers**. This has largely three reasons:

1. Classical architectures were overtaken by Transformers in many areas, especially when data volume is no problem.
2. Principles of why classical architectures work can be much more easily understood by learners. In direct comparisons with Transformers we can thus see where these principles deviate or generalize and thus better understand Transformers.
3. The learner of this course gains the ability to understand a wide variety of model-classes and thus to research additional information on architectures relevant in their respective subfield in a guided manner.

This balancing act comes at the cost of cutting corners in some more theoretically oriented topics and a wide introduction of 
current state-of-the-art architectures.

Remark: The course is still being built. Sessions 1-4 have full slide decks;
Sessions 5-10 will follow in September and October 2026. Content, structure, and
session order may still change.

## Setup

1. Clone `latex-math` into this directory:
   `git clone https://github.com/slds-lmu/latex-math.git latex-math`
   (no license is stated on that repository; it's used only as a build-time
   dependency, cloned locally by each user and gitignored here, never
   redistributed with this repository)
2. Navigate to a session folder, e.g. `session-01-dl-intro-i/`
3. Run `make` to build every `slides-*.tex` deck in that folder to PDF via latexmk

## Structure

Ten sessions, one per teaching day:

1. DL Intro I
2. DL Intro II
3. Backpropagation
4. Optimizers
5. CNN
6. Models for Sequential Data & Generative Models
7. Transformer
8. Reinforcement Learning
9. LLMs
10. Software, Tooling & Hardware

## License

This repository is licensed **CC BY 4.0** (see `LICENSE`), with one
exception: the Neural Architecture Search section in Session 4
(`session-04-optimizers`) is adapted from CC BY-SA licensed material
and is itself shared under **CC BY-SA** terms instead. See that
deck's "License & Attribution" frame for details.

## Acknowledgements

This course would not exist without the openly published teaching materials
of LMU Munich that most of it is built from. Our sincere thanks go
to **Prof. Dr. Bernd Bischl** and the **Statistical Learning and Data
Science (SLDS)** group at LMU Munich, the primary source for this course.
Thanks also to everyone on the wider I2ML, I2DL and DL4NLP teaching teams
who built and maintained that material over the years. I2DL and DL4NLP are
published under **CC BY 4.0**; I2ML is published under the **MIT License**.
Material from all three has been merged, reordered, and adapted for this
course rather than used verbatim. See each session's `order.txt` and the
"License & Attribution" frame at the end of every built deck for the exact
per-section sourcing.

We'd also like to thank Marius Lindauer and Katharina Eggensperger for their
ESSAI 2023 summer school lecture on Neural Architecture Search, published
under **CC BY-SA**, which the NAS section in this course builds on and
shares under those same terms, and David Silver, whose reinforcement-learning
teaching materials will anchor the RL-fundamentals half of Session 8 once
it's built (license not yet confirmed, to be checked before that content is
added). The hands-on tooling segment planned for Session 10 will lean on the
open-source MLflow (**Apache 2.0**), TensorBoard (**Apache 2.0**), and
Hugging Face `transformers`/`datasets` (**Apache 2.0**) projects, plus
Weights & Biases (client SDK **MIT**; the hosted platform itself is a
commercial service, used here only via its free academic tier). Our thanks
to everyone maintaining that tooling for the wider ML community.
