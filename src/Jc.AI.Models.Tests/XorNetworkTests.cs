namespace Jc.AI.Models.Tests;

public class XorNetworkTests
{
    private static XorNetwork.Sample[] CreateSamples() =>
        [new(0, 0, 0), new(0, 1, 1), new(1, 0, 1), new(1, 1, 0)];

    [Test]
    public void Constructor_Initializes17ParametersRepeatablyFromSeed()
    {
        var network = new XorNetwork(42);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(network.Parameters, Has.Count.EqualTo(17));
            Assert.That(network.Parameters, Is.EqualTo(new XorNetwork(42).Parameters));
            Assert.That(network.Parameters, Is.Not.EqualTo(new XorNetwork(7).Parameters));
        }
    }

    [TestCase(42, 0)]
    [TestCase(7, 0)]
    [TestCase(42, 500)]
    public void NumericalGradients_MatchAllAnalyticGradientsWithoutChangingParameters(int seed, int steps)
    {
        var network = new XorNetwork(seed);
        var samples = CreateSamples();
        for (var step = 0; step < steps; step++)
            network.TrainStep(samples, 0.1);
        var before = network.Parameters.ToArray();

        var analytic = network.Evaluate(samples).Gradients;
        var numerical = network.NumericalGradients(samples);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(analytic, Has.Length.EqualTo(17));
            Assert.That(numerical, Is.EqualTo(analytic).Within(1e-6));
            Assert.That(network.Parameters, Is.EqualTo(before));
        }
    }

    [Test]
    public void XorDemo_DefaultSeed_FitsAllFourCasesAfter10000Updates()
    {
        var network = new XorNetwork(42);
        var samples = CreateSamples();
        for (var step = 0; step < 10000; step++)
            network.TrainStep(samples, 0.1);
        var before = network.Parameters.ToArray();

        using (Assert.EnterMultipleScope())
        {
            Assert.That(network.Evaluate(samples).Loss, Is.LessThan(1e-4));
            foreach (var (x1, x2, target) in samples)
                Assert.That(network.Predict(x1, x2) >= 0.5 ? 1 : 0, Is.EqualTo(target), $"({x1}, {x2})");
            Assert.That(network.Parameters, Is.EqualTo(before));
        }
    }

    [Test]
    public void Evaluate_EmptySamples_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => new XorNetwork(42).Evaluate([]));
    }

    [TestCase(double.NaN, 0)]
    [TestCase(double.PositiveInfinity, 0)]
    [TestCase(double.NegativeInfinity, 0)]
    [TestCase(0, double.NaN)]
    [TestCase(0, double.PositiveInfinity)]
    [TestCase(0, double.NegativeInfinity)]
    public void Forward_NonfiniteInput_Throws(double x1, double x2)
    {
        var network = new XorNetwork(42);

        using (Assert.EnterMultipleScope())
        {
            Assert.Throws<InvalidOperationException>(() => network.Predict(x1, x2));
            Assert.Throws<InvalidOperationException>(() => network.Inspect(x1, x2));
            Assert.Throws<InvalidOperationException>(() => network.Evaluate([new(x1, x2, 0)]));
        }
    }

    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    [TestCase(double.NegativeInfinity)]
    public void Evaluate_NonfiniteTarget_Throws(double target)
    {
        Assert.Throws<InvalidOperationException>(() => new XorNetwork(42).Evaluate([new(0, 0, target)]));
    }

    [TestCase(0)]
    [TestCase(-0.1)]
    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    [TestCase(double.NegativeInfinity)]
    public void TrainStep_InvalidLearningRate_Throws(double learningRate)
    {
        Assert.Throws<InvalidOperationException>(() => new XorNetwork(42).TrainStep(CreateSamples(), learningRate));
    }

    [TestCase(0)]
    [TestCase(-1e-6)]
    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    [TestCase(double.NegativeInfinity)]
    public void NumericalGradients_InvalidEpsilon_Throws(double epsilon)
    {
        Assert.Throws<InvalidOperationException>(() => new XorNetwork(42).NumericalGradients(CreateSamples(), epsilon));
    }

    [Test]
    public void TrainStep_OverflowingUpdate_ThrowsWithoutChangingParameters()
    {
        var network = new XorNetwork(42);
        var before = network.Parameters.ToArray();

        Assert.Throws<InvalidOperationException>(() => network.TrainStep([new(0, 0, 10)], double.MaxValue));

        Assert.That(network.Parameters, Is.EqualTo(before));
    }

    [TestCase(0)]
    [TestCase(42)]
    public void NamedSamplesAndResults_PreservePredictionsGradientsAndTraining(int seed)
    {
        var network = new XorNetwork(seed);
        var examples = CreateSamples();
        var parameters = network.Parameters.ToArray();

        var inspection = network.Inspect(0, 1);
        var result = network.Evaluate(examples);
        var numerical = network.NumericalGradients(examples);
        var expectedLoss = examples.Average(example =>
            Math.Pow(network.Predict(example.X1, example.X2) - example.Target, 2));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(inspection.Prediction, Is.EqualTo(network.Predict(0, 1)));
            Assert.That(inspection.Hidden, Has.Length.EqualTo(4));
            Assert.That(result.Loss, Is.EqualTo(expectedLoss).Within(1e-12));
            Assert.That(result.Gradients, Has.Length.EqualTo(parameters.Length));
            Assert.That(numerical, Is.EqualTo(result.Gradients).Within(1e-6));
            Assert.That(network.Parameters, Is.EqualTo(parameters));
        }

        network.TrainStep(examples, learningRate: 0.01);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(network.Parameters,
                Is.EqualTo(parameters.Select((value, i) => value - 0.01 * result.Gradients[i])).Within(1e-12));
            Assert.That(network.Evaluate(examples).Loss, Is.LessThan(result.Loss));
        }
    }
}
