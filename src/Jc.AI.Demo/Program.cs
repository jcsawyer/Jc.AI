using System.Globalization;
using System.Text.Json;
using Jc.AI.Demo;
using Jc.AI.Models;
using Jc.AI.Tokenization;

try
{
    switch (args)
    {
        case []:
        case ["help"]:
        case ["--help"]:
            PrintHelp();
            return 0;
        case ["data"]:
            PrintData();
            return 0;
        case ["tokenize"]:
            PrintTokenization();
            return 0;
        case ["tokenize", var input]:
            PrintTokenization(input);
            return 0;
        case ["bigram"]:
            PrintBigram("Mira", 42);
            return 0;
        case ["bigram", var prompt] when !prompt.StartsWith("--", StringComparison.Ordinal):
            PrintBigram(prompt, 42);
            return 0;
        case ["bigram", "--seed", var bigramSeedText] when
            int.TryParse(bigramSeedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var bigramSeed):
            PrintBigram("Mira", bigramSeed);
            return 0;
        case ["bigram", var seedPrompt, "--seed", var promptSeedText] when
            int.TryParse(promptSeedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var promptSeed):
            PrintBigram(seedPrompt, promptSeed);
            return 0;
        case ["neuron"]:
            PrintNeuron(0.1);
            return 0;
        case ["neuron", "--rate", var rateText] when
            double.TryParse(rateText, NumberStyles.Float, CultureInfo.InvariantCulture, out var rate)
            && double.IsFinite(rate) && rate > 0:
            PrintNeuron(rate);
            return 0;
        case ["xor"]:
            XorDemo.Run(42);
            return 0;
        case ["xor", "--seed", var xorSeedText] when
            int.TryParse(xorSeedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var xorSeed):
            XorDemo.Run(xorSeed);
            return 0;
        default:
            Console.Error.WriteLine("Invalid command or arguments. Use 'help' for usage information.");
            return 1;
    }
}
catch (Exception ex) when (ex is IOException or InvalidOperationException)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}

static void PrintHelp()
{
    Console.WriteLine("""
    Jc.AI.Demo
                      
    Usage: dotnet run --project Jc.AI.Demo [command]
    
    dotnet run -- help                 Show commands
    dotnet run -- data                 Print the bundled text and its statistics
    dotnet run -- tokenize ["TEXT"]    Show character IDs and reconstruct text
    dotnet run -- bigram ["PROMPT"] [--seed N]
                                      Count pairs, score, and generate text
    dotnet run -- neuron [--rate R]    Train one linear neuron (default rate: 0.1)
    dotnet run -- xor [--seed N]       Fit XOR with a 2-4-1 network (default seed: 42)

    Bigram defaults: prompt "Mira", seed 42, 120 new characters.
    Counts come from facts.txt; validation.txt is scored separately.
    The neuron demo uses five numerical examples and 100 gradient-descent updates.
    The XOR demo compares a linear baseline with a network trained for 10000 updates.
    """ );
}

static void PrintData()
{
    var text = Dataset.Load();
    var measurement = Dataset.Measure(text);
    Console.WriteLine(text);
    Console.WriteLine($"Lines: {measurement.Lines}");
    Console.WriteLine($"Words: {measurement.Words}");
    Console.WriteLine($"UTF-16 code units: {measurement.CodeUnits}");
}

static void PrintTokenization(string? input = null)
{
    var corpus = Dataset.Load();
    var tokenizer = new CharacterTokenizer(corpus);
    var text = input ?? corpus;
    var ids = tokenizer.Encode(text);
    var decoded = tokenizer.Decode(ids);

    Console.WriteLine($"Vocabulary: {tokenizer.Vocabulary.Count} UTF-16 code units (from facts.txt)");
    for (var id = 0; id < tokenizer.Vocabulary.Count; id++)
    {
        var character = tokenizer.Vocabulary[id];
        Console.WriteLine($"{id}: {JsonSerializer.Serialize(character.ToString())} (U+{(int)character:X4})");
    }
    Console.WriteLine($"Input: {JsonSerializer.Serialize(text)}");
    Console.WriteLine($"Token IDs: [{string.Join(", ", ids)}]");
    Console.WriteLine($"Decoded: {JsonSerializer.Serialize(decoded)}");
    Console.WriteLine($"Round trip: {(text == decoded ? "SUCCESS" : "FAILED")}");
}

static void PrintBigram(string prompt, int seed)
{
    var trainingText = Dataset.Load();
    var validationText = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "data", "validation.txt"));
    // Assign the split first; neither vocabulary nor counts see validation text.
    var tokenizer = new CharacterTokenizer(trainingText);
    var training = tokenizer.Encode(trainingText);
    var validation = tokenizer.Encode(validationText);
    var promptIds = tokenizer.Encode(prompt);
    if (promptIds.Length == 0)
        throw new InvalidOperationException("The prompt must contain at least one known character.");

    var model = new CountedBigram(tokenizer.Vocabulary.Count, training);
    var trainingLoss = model.Loss(training);
    var validationLoss = model.Loss(validation);
    var previous = promptIds[^1];
    var generated = model.Generate(previous, 120, new Random(seed));

    Console.WriteLine("Phase 3: counted character bigram (add-one smoothing).");
    Console.WriteLine($"Training: stories.txt ({training.Length} tokens, {training.Length - 1} pairs)");
    Console.WriteLine($"Validation: validation.txt ({validation.Length} tokens, {validation.Length - 1} pairs)");
    Console.WriteLine($"Vocabulary: {tokenizer.Vocabulary.Count} UTF-16 code units (training only)");
    Console.WriteLine(FormattableString.Invariant($"Uniform baseline loss: {Math.Log(model.VocabularySize):F6} nats/pair"));
    Console.WriteLine(FormattableString.Invariant($"Training loss: {trainingLoss:F6} nats/pair"));
    Console.WriteLine(FormattableString.Invariant($"Validation loss: {validationLoss:F6} nats/pair"));
    Console.WriteLine($"Next tokens after {JsonSerializer.Serialize(tokenizer.Vocabulary[previous].ToString())} (top 5):");
    foreach (var id in Enumerable.Range(0, model.VocabularySize)
        .OrderByDescending(id => model.Probability(previous, id)).ThenBy(id => id).Take(5))
    {
        var character = JsonSerializer.Serialize(tokenizer.Vocabulary[id].ToString());
        Console.WriteLine(FormattableString.Invariant($"{character}: count={model.Count(previous, id)}, probability={model.Probability(previous, id):F6}"));
    }
    Console.WriteLine($"Seed: {seed}; new tokens: {generated.Length}");
    Console.WriteLine($"Prompt: {JsonSerializer.Serialize(prompt)}");
    Console.WriteLine($"Generated: {JsonSerializer.Serialize(prompt + tokenizer.Decode(generated))}");
}

static void PrintNeuron(double learningRate)
{
    var samples = new LinearNeuron.Sample[] { new(-2, -3), new(-1, -1), new(0, 1), new(1, 3), new(2, 5) };
    var neuron = new LinearNeuron();
    var initial = neuron.Evaluate(samples);
    var numerical = neuron.NumericalGradients(samples);
    if (Math.Abs(initial.WeightGradient - numerical.Weight) > 1e-6 || Math.Abs(initial.BiasGradient - numerical.Bias) > 1e-6)
        throw new InvalidOperationException("The neuron gradient check failed.");

    Console.WriteLine("Phase 4: one linear neuron (prediction = weight * input + bias).");
    Console.WriteLine("Training pairs: (-2, -3), (-1, -1), (0, 1), (1, 3), (2, 5)");
    Console.WriteLine(FormattableString.Invariant($"Learning rate: {learningRate:G}; updates: 100; initial weight: 0; initial bias: 0"));
    Console.WriteLine(FormattableString.Invariant($"Initial analytic gradients: weight={initial.WeightGradient:F6}, bias={initial.BiasGradient:F6}"));
    Console.WriteLine(FormattableString.Invariant($"Initial numerical gradients: weight={numerical.Weight:F6}, bias={numerical.Bias:F6}"));
    Console.WriteLine("Gradient check (epsilon=1E-6, absolute tolerance=1E-6)");
    Console.WriteLine("Step | Weight | Bias | Mean squared error");
    for (var step = 0; step <= 100; step++)
    {
        if (step is 0 or 1 or 10 or 25 or 50 or 100)
        {
            var loss = neuron.Evaluate(samples).Loss;
            Console.WriteLine(FormattableString.Invariant($"{step} | {neuron.Weight:F6} | {neuron.Bias:F6} | {loss:E6}"));
        }
        if (step < 100)
            neuron.TrainStep(samples, learningRate);
    }
    Console.WriteLine(FormattableString.Invariant($"Unseen input: x=3; prediction={neuron.Predict(3):F6}; reference=7"));
    Console.WriteLine("Inference uses the learned parameters without changing them. This demo does not process text.");
}
