namespace Jc.AI.Models.Tests;

public class LinearNeuronTests
{
    private static LinearNeuron.Sample[] CreateSamples() =>
        [new(-2, -3), new(-1, -1), new(0, 1), new(1, 3), new(2, 5)];

    [Test]
    public void Evaluate_DefaultParameters_ReturnsKnownMeanSquaredErrorAndGradients()
    {
        var neuron = new LinearNeuron();

        Assert.That(neuron.Evaluate(CreateSamples()), Is.EqualTo(new LinearNeuron.EvaluationResult(9, -8, -2)));
    }

    [Test]
    public void Evaluate_SingleSample_ReturnsKnownLossAndGradients()
    {
        var neuron = new LinearNeuron();

        Assert.That(neuron.Evaluate([new(2, 5)]), Is.EqualTo(new LinearNeuron.EvaluationResult(25, -20, -10)));
    }

    [Test]
    public void TrainStep_SingleSample_UpdatesBothParametersUsingOldGradients()
    {
        var neuron = new LinearNeuron();

        neuron.TrainStep([new(2, 5)], 0.1);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.EqualTo(2));
            Assert.That(neuron.Bias, Is.EqualTo(1));
        }
    }

    [Test]
    public void Predict_ReturnsKnownForwardPredictionWithoutChangingParameters()
    {
        var neuron = new LinearNeuron(2, 1);

        var prediction = neuron.Predict(3);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(prediction, Is.EqualTo(7));
            Assert.That(neuron.Weight, Is.EqualTo(2));
            Assert.That(neuron.Bias, Is.EqualTo(1));
        }
    }

    [TestCase(0, 0)]
    [TestCase(0.75, -0.25)]
    [TestCase(-1.2, 2.3)]
    public void NumericalGradients_MatchAnalyticGradientsWithoutChangingParameters(double weight, double bias)
    {
        var neuron = new LinearNeuron(weight, bias);
        var samples = CreateSamples();

        var analytic = neuron.Evaluate(samples);
        var numerical = neuron.NumericalGradients(samples);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(numerical.Weight, Is.EqualTo(analytic.WeightGradient).Within(1e-6));
            Assert.That(numerical.Bias, Is.EqualTo(analytic.BiasGradient).Within(1e-6));
            Assert.That(neuron.Weight, Is.EqualTo(weight));
            Assert.That(neuron.Bias, Is.EqualTo(bias));
        }
    }

    [Test]
    public void TrainStep_DefaultRate_ReturnsKnownFirstUpdateAndLoss()
    {
        var neuron = new LinearNeuron();
        var samples = CreateSamples();

        neuron.TrainStep(samples, 0.1);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.EqualTo(0.8).Within(1e-12));
            Assert.That(neuron.Bias, Is.EqualTo(0.2).Within(1e-12));
            Assert.That(neuron.Evaluate(samples).Loss, Is.EqualTo(3.52).Within(1e-12));
        }
    }

    [Test]
    public void NeuronDemo_DefaultRate_ReducesLossAndPredictsUnseenInputAfter100Updates()
    {
        var neuron = new LinearNeuron();
        var samples = CreateSamples();
        var loss = neuron.Evaluate(samples).Loss;

        for (var step = 0; step < 100; step++)
        {
            neuron.TrainStep(samples, 0.1);
            var nextLoss = neuron.Evaluate(samples).Loss;
            Assert.That(nextLoss, Is.LessThanOrEqualTo(loss + 1e-15), $"Step {step + 1}");
            loss = nextLoss;
        }

        var weight = neuron.Weight;
        var bias = neuron.Bias;
        var prediction = neuron.Predict(3);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(loss, Is.LessThan(1e-15));
            Assert.That(prediction, Is.EqualTo(7).Within(1e-9));
            Assert.That(neuron.Weight, Is.EqualTo(weight));
            Assert.That(neuron.Bias, Is.EqualTo(bias));
        }
    }

    [Test]
    public void NeuronDemo_DisplayedSteps_MatchClosedFormTraining(
        [Values(0.1, 0.05, 0.2)] double learningRate,
        [Values(0, 1, 10, 25, 50, 100)] int step)
    {
        var neuron = new LinearNeuron();
        var samples = CreateSamples();
        for (var i = 0; i < step; i++)
            neuron.TrainStep(samples, learningRate);

        // For this symmetric dataset, MSE = 2(w - 2)^2 + (b - 1)^2.
        // Each update scales the weight error by 1 - 4r and the bias error by 1 - 2r.
        var weightError = -2 * Math.Pow(1 - 4 * learningRate, step);
        var biasError = -Math.Pow(1 - 2 * learningRate, step);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.EqualTo(2 + weightError).Within(1e-12));
            Assert.That(neuron.Bias, Is.EqualTo(1 + biasError).Within(1e-12));
            Assert.That(neuron.Evaluate(samples).Loss,
                Is.EqualTo(2 * weightError * weightError + biasError * biasError).Within(1e-12));
            Assert.That(neuron.Predict(3), Is.EqualTo(7 + 3 * weightError + biasError).Within(1e-12));
        }
    }

    [TestCase(double.NaN, 0)]
    [TestCase(double.PositiveInfinity, 0)]
    [TestCase(0, double.NegativeInfinity)]
    [TestCase(0, double.NaN)]
    public void Constructor_NonfiniteParameters_Throws(double weight, double bias)
    {
        Assert.Throws<InvalidOperationException>(() => _ = new LinearNeuron(weight, bias));
    }

    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    [TestCase(double.NegativeInfinity)]
    public void Predict_NonfiniteInput_Throws(double input)
    {
        Assert.Throws<InvalidOperationException>(() => new LinearNeuron().Predict(input));
    }

    [Test]
    public void Evaluate_EmptySamples_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => new LinearNeuron().Evaluate([]));
    }

    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    [TestCase(double.NegativeInfinity)]
    public void Evaluate_NonfiniteTarget_Throws(double target)
    {
        Assert.Throws<InvalidOperationException>(() => new LinearNeuron().Evaluate([new(0, target)]));
    }

    [TestCase(0)]
    [TestCase(-0.1)]
    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    public void TrainStep_InvalidLearningRate_Throws(double learningRate)
    {
        Assert.Throws<InvalidOperationException>(() => new LinearNeuron().TrainStep(CreateSamples(), learningRate));
    }

    [TestCase(0)]
    [TestCase(-1e-6)]
    [TestCase(double.NaN)]
    [TestCase(double.PositiveInfinity)]
    public void NumericalGradients_InvalidEpsilon_Throws(double epsilon)
    {
        Assert.Throws<InvalidOperationException>(() => new LinearNeuron().NumericalGradients(CreateSamples(), epsilon));
    }

    [Test]
    public void TrainStep_OverflowingUpdate_ThrowsWithoutChangingParameters()
    {
        var neuron = new LinearNeuron();

        Assert.Throws<InvalidOperationException>(() => neuron.TrainStep(CreateSamples(), double.MaxValue));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.Zero);
            Assert.That(neuron.Bias, Is.Zero);
        }
    }

    [TestCase(double.MaxValue, 0)]
    [TestCase(0, double.MaxValue)]
    public void NumericalGradients_OverflowingPerturbation_ThrowsWithoutChangingParameters(double weight, double bias)
    {
        var neuron = new LinearNeuron(weight, bias);

        Assert.Throws<InvalidOperationException>(() =>
            neuron.NumericalGradients([new(0, bias)], epsilon: double.MaxValue));

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.EqualTo(weight));
            Assert.That(neuron.Bias, Is.EqualTo(bias));
        }
    }

    [Test]
    public void EvaluationAndTraining_UseNamedExamplesAndResults()
    {
        var neuron = new LinearNeuron(weight: 1, bias: 0.5);
        LinearNeuron.Sample[] examples = [new(1, 3), new(2, 5)];

        var result = neuron.Evaluate(examples);
        var gradients = neuron.NumericalGradients(examples);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(result.Loss, Is.EqualTo(4.25).Within(1e-12));
            Assert.That(result.WeightGradient, Is.EqualTo(-6.5).Within(1e-12));
            Assert.That(result.BiasGradient, Is.EqualTo(-4).Within(1e-12));
            Assert.That(gradients.Weight, Is.EqualTo(result.WeightGradient).Within(1e-8));
            Assert.That(gradients.Bias, Is.EqualTo(result.BiasGradient).Within(1e-8));
            Assert.That(neuron.Weight, Is.EqualTo(1));
            Assert.That(neuron.Bias, Is.EqualTo(0.5));
        }

        neuron.TrainStep(examples, learningRate: 0.1);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(neuron.Weight, Is.EqualTo(1.65).Within(1e-12));
            Assert.That(neuron.Bias, Is.EqualTo(0.9).Within(1e-12));
            Assert.That(neuron.Evaluate(examples).Loss, Is.LessThan(result.Loss));
        }
    }
}
