namespace Jc.AI.Models;

public sealed class LinearNeuron
{
    public readonly record struct Sample(double Input, double Target);
    public readonly record struct EvaluationResult(double Loss, double WeightGradient, double BiasGradient);
    public readonly record struct Gradients(double Weight, double Bias);

    public double Weight { get; private set; }
    public double Bias { get; private set; }

    public LinearNeuron(double weight = 0, double bias = 0)
    {
        if (!double.IsFinite(weight) || !double.IsFinite(bias))
        {
            throw new InvalidOperationException("Neuron parameters must be finite.");
        }
        Weight = weight;
        Bias = bias;
    }
    
    public double Predict(double input) => Predict(input, Weight, Bias);

    private static double Predict(double input, double weight, double bias)
    {
        var prediction = weight * input + bias;
        if (!double.IsFinite(input) || !double.IsFinite(prediction))
        {
            throw new InvalidOperationException("Neuron input and prediction must be finite.");
        }
        return prediction;
    }
    
    public EvaluationResult Evaluate(Sample[] samples) => Evaluate(samples, Weight, Bias);

    private static EvaluationResult Evaluate(Sample[] samples, double weight, double bias)
    {
        if (!double.IsFinite(weight) || !double.IsFinite(bias))
        {
            throw new InvalidOperationException("Neuron parameters must be finite.");
        }

        if (samples.Length == 0)
        {
            throw new InvalidOperationException("Neuron evaluation needs at least one example.");
        }

        var loss = 0.0;
        var weightGradient = 0.0;
        var biasGradient = 0.0;
        foreach (var example in samples)
        {
            if (!double.IsFinite(example.Target))
            {
                throw new InvalidOperationException("Neuron targets must be finite.");
            }
            var error = Predict(example.Input, weight, bias) - example.Target;
            loss += error * error;
            weightGradient += 2 * error * example.Input;
            biasGradient += 2 * error;
        }
        loss /= samples.Length;
        weightGradient /= samples.Length;
        biasGradient /= samples.Length;
        if (!double.IsFinite(loss) || !double.IsFinite(weightGradient) || !double.IsFinite(biasGradient))
        {
            throw new InvalidOperationException(
                "Neuron loss or gradient overflowed; use smaller values or a smaller learning rate.");
        }
        return new EvaluationResult(loss, weightGradient, biasGradient);
    }

    public void TrainStep(Sample[] samples, double learningRate)
    {
        if (!double.IsFinite(learningRate) || learningRate <= 0)
        {
            throw new InvalidOperationException("The learning rate must be positive and finite.");
        }

        var result = Evaluate(samples);
        var weight = Weight - learningRate * result.WeightGradient;
        var bias = Bias - learningRate * result.BiasGradient;
        if (!double.IsFinite(weight) || !double.IsFinite(bias))
        {
            throw new InvalidOperationException("Neuron update overflowed; use a smaller learning rate.");
        }
        Weight = weight;
        Bias = bias;
    }

    public Gradients NumericalGradients(Sample[] samples, double epsilon = 1e-6)
    {
        if (!double.IsFinite(epsilon) || epsilon <= 0)
        {
            throw new InvalidOperationException("The finite-difference epsilon must be positive and finite.");
        }

        var weightPlus = Evaluate(samples, Weight + epsilon, Bias).Loss;
        var weightMinus = Evaluate(samples, Weight - epsilon, Bias).Loss;
        var biasPlus = Evaluate(samples, Weight, Bias + epsilon).Loss;
        var biasMinus = Evaluate(samples, Weight, Bias - epsilon).Loss;
        return new Gradients((weightPlus - weightMinus) / (2 * epsilon), (biasPlus - biasMinus) / (2 * epsilon));
    }
}
