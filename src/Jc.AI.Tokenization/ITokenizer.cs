namespace Jc.AI.Tokenization;

public interface ITokenizer
{
    int[] Encode(string text);
    string Decode(IEnumerable<int> tokenIds);
}