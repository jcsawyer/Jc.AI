using System.Text.Json;
using Jc.AI.Demo;
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
                                      
    Bigram defaults: prompt "Mira", seed 42, 120 new characters.
    Counts come from facts.txt; validation.txt is scored separately.
    The neuron demo uses five numerical examples and 100 gradient-descent updates.
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
