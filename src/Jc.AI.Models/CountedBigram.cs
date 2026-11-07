namespace Jc.AI.Models;

public sealed class CountedBigram
{
    private readonly int[,] _counts;
    private readonly int[] _totals;
    internal int VocabularySize => _totals.Length;

    public CountedBigram(int vocabularySize, int[] trainingTokens)
    {
        if (vocabularySize < 1 || trainingTokens.Length < 2)
        {
            throw new InvalidOperationException(
                "A bigram model needs a nonempty vocabulary and at least two training tokens.");
        }

        _counts = new int[vocabularySize, vocabularySize];
        _totals = new int[vocabularySize];
        foreach (var id in trainingTokens)
        {
            ValidateId(id);
        }

        for (var i = 1; i < trainingTokens.Length; i++)
        {
            var previous = trainingTokens[i - 1];
            var next = trainingTokens[i];
            _counts[previous, next]++;
            _totals[previous]++;
        }
    }
    
    public int Count(int previous, int next)
    {
        ValidateId(previous);
        ValidateId(next);
        return _counts[previous, next];
    }
    
    public double Probability(int previous, int next) =>
        (Count(previous, next) + 1.0) / (_totals[previous] + (double)VocabularySize);
    
    public double Loss(int[] tokens)
    {
        if (tokens.Length < 2)
        {
            throw new InvalidOperationException("Scoring needs at least two tokens (one adjacent pair).");
        }
        
        var loss = 0.0;
        for (var i = 1; i < tokens.Length; i++)
        {
            loss -= Math.Log(Probability(tokens[i - 1], tokens[i]));
        }
        return loss / (tokens.Length - 1);
    }
    
    public int[] Generate(int previous, int newTokens, Random random)
    {
        ValidateId(previous);
        if (newTokens < 0)
        {
            throw new InvalidOperationException("The number of new tokens cannot be negative.");
        }

        var result = new int[newTokens];
        for (var i = 0; i < newTokens; i++)
        {
            var draw = random.NextDouble();
            var cumulative = 0.0;
            for (var next = 0; next < VocabularySize; next++)
            {
                cumulative += Probability(previous, next);
                if (draw < cumulative)
                {
                    result[i] = next;
                    previous = next;
                    break;
                }
            }
        }
        return result;
    }

    private void ValidateId(int id)
    {
        if (id < 0 || id >= VocabularySize)
            throw new InvalidOperationException($"Token ID {id} is outside the bigram vocabulary (size {VocabularySize}).");
    }
}