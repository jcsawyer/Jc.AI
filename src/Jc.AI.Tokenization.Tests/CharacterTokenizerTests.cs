namespace Jc.AI.Tokenization.Tests;

public class CharacterTokenizerTests
{
    private static string ReadFacts() =>
        File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "data", "facts.txt"));

    [Test]
    public void Vocabulary_IsDistinctAndSortedByCodeUnit()
    {
        var tokenizer = new CharacterTokenizer("cabca");

        Assert.That(tokenizer.Vocabulary, Is.EqualTo(['a', 'b', 'c']));
    }

    [Test]
    public void Encode_Cabca_ReturnsHandWorkedIds()
    {
        var tokenizer = new CharacterTokenizer("cabca");

        Assert.That(tokenizer.Encode("cabca"), Is.EqualTo([2, 0, 1, 2, 0]));
    }

    [Test]
    public void Decode_HandWorkedIds_ReturnsCabca()
    {
        var tokenizer = new CharacterTokenizer("cabca");

        Assert.That(tokenizer.Decode([2, 0, 1, 2, 0]), Is.EqualTo("cabca"));
    }

    [Test]
    public void FactsVocabulary_MatchesDocumentedCodeUnitsAndIds()
    {
        var tokenizer = new CharacterTokenizer(ReadFacts());

        using (Assert.EnterMultipleScope())
        {
            Assert.That(tokenizer.Vocabulary, Is.EqualTo("\n .LMNPSTabcdeghiklmnoprstuvw".ToCharArray()));
            Assert.That(tokenizer.Encode("Mira"), Is.EqualTo([4, 16, 23, 9]));
            Assert.That(tokenizer.Decode([4, 16, 23, 9]), Is.EqualTo("Mira"));
        }
    }

    [Test]
    public void Facts_RoundTripExactly()
    {
        var facts = ReadFacts();
        var tokenizer = new CharacterTokenizer(facts);

        var ids = tokenizer.Encode(facts);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(ids, Has.Length.EqualTo(facts.Length));
            Assert.That(tokenizer.Decode(ids), Is.EqualTo(facts));
        }
    }

    [TestCase("")]
    [TestCase(" \t\r\n  \n")]
    [TestCase("aAbBcC")]
    [TestCase("Hello, world.!?;:'\"-()")]
    [TestCase("café naïve")]
    [TestCase("e\u0301")]
    [TestCase("A😀B")]
    public void Text_RoundTripsExactly(string text)
    {
        var tokenizer = new CharacterTokenizer(text);

        var ids = tokenizer.Encode(text);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(ids, Has.Length.EqualTo(text.Length));
            Assert.That(tokenizer.Decode(ids), Is.EqualTo(text));
        }
    }

    [Test]
    public void EmptyText_EncodesAndDecodesWithNonEmptyVocabulary()
    {
        var tokenizer = new CharacterTokenizer("abc");

        using (Assert.EnterMultipleScope())
        {
            Assert.That(tokenizer.Encode(""), Is.Empty);
            Assert.That(tokenizer.Decode([]), Is.EqualTo(""));
        }
    }

    [Test]
    public void Emoji_UsesTwoUtf16CodeUnits()
    {
        var tokenizer = new CharacterTokenizer("😀");

        using (Assert.EnterMultipleScope())
        {
            Assert.That(tokenizer.Vocabulary, Is.EqualTo(['\uD83D', '\uDE00']));
            Assert.That(tokenizer.Encode("😀"), Is.EqualTo([0, 1]));
            Assert.That(tokenizer.Decode([0, 1]), Is.EqualTo("😀"));
        }
    }

    [TestCase("Z", "U+005A")]
    [TestCase("a\u0301", "U+0301")]
    [TestCase("😀", "U+D83D")]
    public void Encode_UnknownCodeUnit_ThrowsWithoutChangingVocabulary(string text, string codeUnit)
    {
        var tokenizer = new CharacterTokenizer("abc");

        var exception = Assert.Throws<InvalidOperationException>(() => tokenizer.Encode(text));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(exception!.Message, Does.Contain(codeUnit).And.Contain("absent"));
            Assert.That(tokenizer.Vocabulary, Is.EqualTo(['a', 'b', 'c']));
        }
    }

    [Test]
    public void Encode_Z_IsAbsentFromFactsVocabulary()
    {
        var tokenizer = new CharacterTokenizer(ReadFacts());

        var exception = Assert.Throws<InvalidOperationException>(() => tokenizer.Encode("MiraZ"));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(exception!.Message, Does.Contain("U+005A").And.Contain("absent"));
            Assert.That(tokenizer.Vocabulary, Has.Count.EqualTo(29));
        }
    }

    [TestCase("abc", -1)]
    [TestCase("abc", 3)]
    [TestCase("abc", int.MinValue)]
    [TestCase("abc", int.MaxValue)]
    [TestCase("", -1)]
    [TestCase("", 0)]
    public void Decode_InvalidId_ThrowsWithIdAndVocabularySize(string vocabulary, int id)
    {
        var tokenizer = new CharacterTokenizer(vocabulary);

        var exception = Assert.Throws<InvalidOperationException>(() => tokenizer.Decode([id]));

        Assert.That(exception!.Message,
            Does.Contain($"Token ID {id} ").And.Contain($"size {vocabulary.Length}"));
    }
}
