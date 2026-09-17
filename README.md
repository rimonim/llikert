# LLikert

**Rate texts with a language model, and get probabilities instead of a single answer.**

LLikert is for researchers who want a language model to code or rate text data: open-ended survey answers, diary entries, interview excerpts, social media posts. Think of a coding scheme ("Is this a question, a request, or a description?") or a rating scale ("How positive is this, from 1 to 5?"). It can also have the model answer questionnaire items itself.

A chatbot gives you one answer per text. LLikert instead reports **how likely the model considered each possible answer**. For every text you get something like:

| id | description | question | request |
|---|---|---|---|
| p01 | 0.02 | 0.95 | 0.03 |
| p02 | 0.48 | 0.04 | 0.48 |

The first text is clearly a question. For the second, the model is torn between two categories, which is useful information you would lose with a single answer. For rating scales, LLikert can also give an expected score, such as 3.7 on a 1–5 scale.

It works from **R** or **Python**, on datasets of any size, and it can pick up where it left off if your computer or connection is interrupted.

## How it works, briefly

1. You write short instructions and list your categories, each with a one-character answer code such as `A`, `B`, `C`.
2. For each item (a text, or a questionnaire statement), the model reads your instructions, the list of codes, and the item.
3. Instead of letting the model write an answer, LLikert looks at the probability the model gives to each of your answer codes as its very next word.

You can see exactly what the model reads before scoring anything, and change its wording and layout. See [`docs/prompts.md`](docs/prompts.md).

The language model runs on a separate computer with a powerful graphics card, called the **scoring service**. Usually one person in a lab (or the lab's IT support) sets it up once. Everyone else only needs R or Python, the service's web address, and an access key.

## Getting started

**You need:**
- **The service address and an access key** from whoever runs your lab's scoring service. If nobody does yet, point them to [`deploy/README.md`](deploy/README.md).
- **R (version 4.1 or newer) or Python (3.10 or newer)** on your own computer. No graphics card, model download or special software is needed.

**Install the R package** from GitHub:

```r
install.packages("remotes")
remotes::install_github("rimonim/llikert", subdir = "r/llikert")
```

**Or install the Python package** from GitHub:

```bash
pip install "llikert @ git+https://github.com/rimonim/llikert.git#subdirectory=python"
```

**Then follow the step-by-step guide for your language:**
- R: [`r/llikert/README.md`](r/llikert/README.md)
- Python: [`python/README.md`](python/README.md)

## Before you use the numbers in a study

- **These are the model's probabilities, not the truth.** A probability of 0.9 means the model strongly favored that answer, not that the answer is 90% likely to be correct. Check the scores against human ratings on a sample of your data before relying on them.
- **Small changes can change the results.** Different instructions, prompt layout, category labels, answer codes, order of categories or model can shift the scores. Decide these before scoring your main data, and report them in your paper.
- **LLikert records what you used.** Every saved result includes your task and its prompt format, the answer codes and the exact model and settings, so you can report and reproduce them.
- **Privacy:** your items are sent to the scoring service. If that service runs outside your institution (for example on a cloud provider), check that your ethics approval and data agreements allow it. LLikert does not store texts on the service or in its log files.

## For the person running the service

- [`deploy/README.md`](deploy/README.md): running the service in a container on a GPU machine
- [`docs/local-service.md`](docs/local-service.md): installing it directly on a machine
- [`deploy/hf-endpoint.md`](deploy/hf-endpoint.md): hosting on Hugging Face (experimental, not yet tested)
- [`docs/troubleshooting.md`](docs/troubleshooting.md): common problems

## Technical documentation

- [`docs/compatibility.md`](docs/compatibility.md): which models, graphics cards and settings are verified
- [`docs/benchmark-report.md`](docs/benchmark-report.md): speed and accuracy measurements
- [`docs/protocol.md`](docs/protocol.md) and [`docs/checkpoint-format.md`](docs/checkpoint-format.md): file and network formats
- [`docs/decisions/`](docs/decisions/): design decisions and the evidence behind them
- [`llikert_development_plan.md`](llikert_development_plan.md): the original specification

**Status:** early development (version 0.1 in progress). The software works and is tested on one verified setup. It has not yet been released or validated in a published study.

License: MIT.
