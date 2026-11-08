# Jc.AI

Adventures in AI: neural networks, machine learning, and language models, explored through small, hands-on projects in C# and .NET.

## Getting started

Run these .NET commands from the repository root:

```sh
dotnet build src/Jc.AI.slnx
dotnet test src/Jc.AI.slnx --no-build
```

The NUnit tests cover the hand-worked examples, the full dataset, empty text, whitespace, casing, punctuation, accents, combining marks, emoji, and invalid input. Pull requests also run a Release build and the tests through GitHub Actions.

Bigram tests cover pair counts, smoothing, normalized probabilities, loss, repeatable sampling, invalid input, and the training/validation workflow.

## Building a Tiny GPT - Part 1

This repository follows my journey of building a tiny GPT.

- [Building a Tiny GPT - Part 1: Character Tokenization](https://jcsawyer.com/blog/2026-11-06-building-a-tiny-gpt-part-1)

### Character Tokenizer

The character tokenizer builds a vocabulary of distinct, sorted UTF-16 code units from supplied text, encodes text into token IDs, and decodes those IDs back into the original text. Unknown code units and invalid token IDs produce explicit errors.

### Datasets

The bundled dataset in `data/facts.txt` produces a vocabulary of 29 code units. With that vocabulary, `Mira` encodes to `[4, 16, 23, 9]` and decodes back to `Mira`.

## Building a Tiny GPT - Part 2

- [Building a Tiny GPT - Part 2](https://jcsawyer.com/blog/2026-11-07-building-a-tiny-gpt-part-2)

### Counted Bigram Model

The counted bigram model learns how often each token follows another (at this point, a token is a single character), then uses those counts with add-one smoothing to predict and generate the next character. It considers only the previous token, so generated text captures local character patterns but has little word or sentence structure.

The vocabulary and counts are trained on the `data/facts.txt` from part 1. The additional text in `data/validation.txt` measures how well the model predicts unseen text without training on it. Loss is the average negative log-probability of each next token - lower is better. A uniform baseline provides a comparison with guessing every token equally.

## Building a Tiny GPT - Part 3

- [Building a Tiny GPT - Part 3](https://jcsawyer.com/blog/2026-11-08-building-a-tiny-gpt-part-3)

### Learning with a Linear Neuron

Part 3 introduces a linear neuron that predicts `weight * input + bias`. Starting from zero, it learns the rule `y = 2x + 1` from five numerical examples by adjusting its weight and bias with gradient descent to reduce mean squared error.

The demo checks the calculated gradients against numerical estimates, shows progress over 100 updates, and predicts `7` for the unseen input `3`. Tests cover the gradients, parameter updates, convergence, and invalid input.
