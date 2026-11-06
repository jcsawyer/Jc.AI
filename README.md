# Jc.AI

Adventures in AI: neural networks, machine learning, and language models, explored through small, hands-on projects in C# and .NET.

## Building a Tiny GPT - Part 1

This repository follows my journey of building a tiny GPT.

- [Building a Tiny GPT - Part 1: Character Tokenization](https://jcsawyer.com/blog/2026-11-06-building-a-tiny-gpt-part-1)

### Character Tokenizer

The character tokenizer builds a vocabulary of distinct, sorted UTF-16 code units from supplied text, encodes text into token IDs, and decodes those IDs back into the original text. Unknown code units and invalid token IDs produce explicit errors.

### Datasets

The bundled dataset in `data/facts.txt` produces a vocabulary of 29 code units. With that vocabulary, `Mira` encodes to `[4, 16, 23, 9]` and decodes back to `Mira`.

## Getting started

Run these .NET commands from the repository root:

```sh
dotnet build src/Jc.AI.slnx
dotnet test src/Jc.AI.slnx --no-build
```

The NUnit tests cover the hand-worked examples, the full dataset, empty text, whitespace, casing, punctuation, accents, combining marks, emoji, and invalid input. Pull requests also run a Release build and the tests through GitHub Actions.
