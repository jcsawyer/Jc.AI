using System.Text;

namespace Jc.AI.Tokenization;

public sealed class CharacterTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _tokenIds = new();
    internal IReadOnlyList<char> Vocabulary { get; }

    public CharacterTokenizer(string vocabularyText)
    {
        var characters = vocabularyText.Distinct().Order().ToArray();
        Vocabulary = Array.AsReadOnly(characters);
        for (var id = 0; id < characters.Length; id++)
        {
            _tokenIds.Add(characters[id], id);
        }
    }
    
    public int[] Encode(string text)
    {
        var ids = new int[text.Length];
        for (var i = 0; i < text.Length; i++)
        {
            if (!_tokenIds.TryGetValue(text[i], out var id))
            {
                throw new InvalidOperationException(
                    $"Unknown UTF-16 code unit U+{(int)text[i]:X4}; it is absent from the tokenizer vocabulary.");
            }
            ids[i] = id;
        }
        return ids;
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        var text = new StringBuilder();
        foreach (var id in tokenIds)
        {
            if (id < 0 || id >= Vocabulary.Count)
            {
                throw new InvalidOperationException(
                    $"Token ID {id} is outside the vocabulary (size {Vocabulary.Count}).");
            }
            text.Append(Vocabulary[id]);
        }
        return text.ToString();
    }
}