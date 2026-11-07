using Jc.AI.Tokenization;

namespace Jc.AI.Models.Tests;

public class CountedBigramTests
{
    private static CountedBigram CreateModel() => new(3, [0, 1, 0, 1, 0]);

    [Test]
    public void FactsAndValidation_ScoreHeldOutTextAndGenerateFromPrompt()
    {
        var dataDirectory = Path.Combine(TestContext.CurrentContext.TestDirectory, "data");
        var trainingText = File.ReadAllText(Path.Combine(dataDirectory, "facts.txt"));
        var validationText = File.ReadAllText(Path.Combine(dataDirectory, "validation.txt"));
        var tokenizer = new CharacterTokenizer(trainingText);
        var training = tokenizer.Encode(trainingText);
        var validation = tokenizer.Encode(validationText);
        var prompt = tokenizer.Encode("Mira");
        var vocabularySize = trainingText.Distinct().Count();
        var model = new CountedBigram(vocabularySize, training);
        var pairs = Enumerable.Range(0, vocabularySize)
            .SelectMany(previous => Enumerable.Range(0, vocabularySize)
                .Select(next => (Previous: previous, Next: next))).ToArray();
        var countsBefore = pairs.Select(pair => model.Count(pair.Previous, pair.Next)).ToArray();

        var baselineLoss = Math.Log(model.VocabularySize);
        var trainingLoss = model.Loss(training);
        var validationLoss = model.Loss(validation);
        var generated = model.Generate(prompt[^1], 120, new Random(42));
        var decoded = tokenizer.Decode(generated);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(vocabularySize, Is.EqualTo(29));
            Assert.That(tokenizer.Decode(validation), Is.EqualTo(validationText));
            // Expected losses are calculated from the bundled files using add-one pair counts.
            Assert.That(trainingLoss, Is.EqualTo(2.420321165225579).Within(1e-12));
            Assert.That(validationLoss, Is.EqualTo(2.632872001673284).Within(1e-12));
            Assert.That(trainingLoss, Is.LessThan(baselineLoss));
            Assert.That(validationLoss, Is.LessThan(baselineLoss));
            Assert.That(generated, Has.Length.EqualTo(120));
            Assert.That(generated, Is.All.InRange(0, vocabularySize - 1));
            Assert.That(tokenizer.Encode(decoded), Is.EqualTo(generated));
            Assert.That(generated, Is.EqualTo(model.Generate(prompt[^1], 120, new Random(42))));
            Assert.That(pairs.Select(pair => model.Count(pair.Previous, pair.Next)), Is.EqualTo(countsBefore));
        }
    }

    [TestCase(0, 1, 2)]
    [TestCase(1, 0, 2)]
    [TestCase(0, 0, 0)]
    public void Count_ReturnsAdjacentPairCount(int previous, int next, int expected)
    {
        var model = CreateModel();

        Assert.That(model.Count(previous, next), Is.EqualTo(expected));
    }

    [Test]
    public void Probability_UsesAddOneSmoothing()
    {
        var model = CreateModel();

        Assert.That(model.Probability(0, 1), Is.EqualTo(3.0 / 5).Within(1e-12));
    }

    [TestCase(0)]
    [TestCase(1)]
    [TestCase(2)]
    public void Probability_EachRowSumsToOne(int previous)
    {
        var model = CreateModel();

        var sum = Enumerable.Range(0, model.VocabularySize).Sum(next => model.Probability(previous, next));

        Assert.That(sum, Is.EqualTo(1).Within(1e-12));
    }

    [TestCase(0)]
    [TestCase(1)]
    [TestCase(2)]
    public void Probability_UnseenContext_IsUniform(int next)
    {
        var model = CreateModel();

        Assert.That(model.Probability(2, next), Is.EqualTo(1.0 / 3).Within(1e-12));
    }

    [Test]
    public void Loss_ReturnsMeanNegativeLogLikelihood()
    {
        var model = CreateModel();

        Assert.That(model.Loss([0, 1, 0]), Is.EqualTo(-Math.Log(3.0 / 5)).Within(1e-12));
    }

    [Test]
    public void Loss_UnseenPairs_IsFinite()
    {
        var model = CreateModel();

        Assert.That(double.IsFinite(model.Loss([0, 0, 2])), Is.True);
    }

    [Test]
    public void Generate_SameSeed_RepeatsSequence()
    {
        var model = CreateModel();

        var generated = model.Generate(0, 100, new Random(42));

        Assert.That(generated, Is.EqualTo(model.Generate(0, 100, new Random(42))));
    }

    [Test]
    public void Generate_ReturnsRequestedLengthAndValidIds()
    {
        var model = CreateModel();

        var generated = model.Generate(0, 100, new Random(42));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(generated, Has.Length.EqualTo(100));
            Assert.That(generated, Is.All.InRange(0, model.VocabularySize - 1));
        }
    }

    [Test]
    public void Generate_ZeroNewTokens_ReturnsEmptyArray()
    {
        var model = CreateModel();

        Assert.That(model.Generate(0, 0, new Random(42)), Is.Empty);
    }

    [Test]
    public void Generate_SamplesFollowContextProbabilityDistribution()
    {
        var model = CreateModel();
        var samples = new int[model.VocabularySize];
        var random = new Random(42);

        for (var i = 0; i < 10000; i++)
        {
            samples[model.Generate(0, 1, random)[0]]++;
        }

        using (Assert.EnterMultipleScope())
        {
            for (var id = 0; id < samples.Length; id++)
            {
                Assert.That(samples[id] / 10000.0,
                    Is.EqualTo(model.Probability(0, id)).Within(0.02), $"Token ID {id}");
            }
        }
    }

    [Test]
    public void LossAndGenerate_DoNotUpdateCounts()
    {
        var model = CreateModel();
        var counts = Enumerable.Range(0, model.VocabularySize)
            .SelectMany(previous => Enumerable.Range(0, model.VocabularySize)
                .Select(next => model.Count(previous, next))).ToArray();

        model.Loss([0, 0, 2]);
        model.Generate(0, 100, new Random(42));

        var after = Enumerable.Range(0, model.VocabularySize)
            .SelectMany(previous => Enumerable.Range(0, model.VocabularySize)
                .Select(next => model.Count(previous, next))).ToArray();
        Assert.That(after, Is.EqualTo(counts));
    }

    [TestCase(0, new[] { 0, 0 })]
    [TestCase(-1, new[] { 0, 0 })]
    [TestCase(2, new int[0])]
    [TestCase(2, new[] { 0 })]
    [TestCase(2, new[] { 0, 2 })]
    [TestCase(2, new[] { -1, 0 })]
    public void Constructor_InvalidInput_Throws(int vocabularySize, int[] trainingTokens)
    {
        Assert.Throws<InvalidOperationException>(() => _ = new CountedBigram(vocabularySize, trainingTokens));
    }

    [TestCase(-1, 0)]
    [TestCase(3, 0)]
    [TestCase(0, -1)]
    [TestCase(0, 3)]
    public void CountAndProbability_InvalidId_Throw(int previous, int next)
    {
        var model = CreateModel();

        using (Assert.EnterMultipleScope())
        {
            Assert.Throws<InvalidOperationException>(() => model.Count(previous, next));
            Assert.Throws<InvalidOperationException>(() => model.Probability(previous, next));
        }
    }

    [TestCase(new int[0])]
    [TestCase(new[] { 0 })]
    [TestCase(new[] { 0, 3 })]
    [TestCase(new[] { -1, 0 })]
    public void Loss_InvalidInput_Throws(int[] tokens)
    {
        var model = CreateModel();

        Assert.Throws<InvalidOperationException>(() => model.Loss(tokens));
    }

    [TestCase(-1, 1)]
    [TestCase(3, 1)]
    [TestCase(0, -1)]
    public void Generate_InvalidInput_Throws(int previous, int newTokens)
    {
        var model = CreateModel();

        Assert.Throws<InvalidOperationException>(() => model.Generate(previous, newTokens, new Random(42)));
    }
}
