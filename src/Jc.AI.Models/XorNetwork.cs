namespace Jc.AI.Models;

public sealed class XorNetwork
{
    public readonly record struct Sample(double X1, double X2, double Target);
    public readonly record struct ForwardResult(double Prediction, double[] Hidden);
    public readonly record struct EvaluationResult(double Loss, double[] Gradients);

    private const int HiddenUnits = 4;
    // Each hidden unit has two input weights and a bias (0..11), then output weights (12..15) and bias (16).
    private readonly double[] _parameters = new double[17];
    public IReadOnlyList<double> Parameters { get; }

    public XorNetwork(int seed)
    {
        var random = new Random(seed);
        for (var unit = 0; unit < HiddenUnits; unit++)
        {
            _parameters[3 * unit] = 2 * random.NextDouble() - 1;
            _parameters[3 * unit + 1] = 2 * random.NextDouble() - 1;
            _parameters[12 + unit] = 2 * random.NextDouble() - 1;
        }
        Parameters = _parameters.AsReadOnly();
    }
    
    private XorNetwork(XorNetwork source)
    {
        Array.Copy(source._parameters, _parameters, _parameters.Length);
        Parameters = _parameters.AsReadOnly();
    }
    
    private ForwardResult Forward(double x1, double x2)
    {
        if (!double.IsFinite(x1) || !double.IsFinite(x2))
        {
            throw new InvalidOperationException("XOR inputs must be finite.");
        }

        var hidden = new double[HiddenUnits];
        var prediction = _parameters[16];
        for (var unit = 0; unit < HiddenUnits; unit++)
        {
            var offset = 3 * unit;
            var sum = _parameters[offset] * x1 + _parameters[offset + 1] * x2 + _parameters[offset + 2];
            if (!double.IsFinite(sum))
            {
                throw new InvalidOperationException("XOR hidden preactivation overflowed.");
            }

            hidden[unit] = Math.Tanh(sum);
            prediction += _parameters[12 + unit] * hidden[unit];
        }
        if (!double.IsFinite(prediction))
        {
            throw new InvalidOperationException("XOR prediction overflowed.");
        }

        return new ForwardResult(prediction, hidden);
    }

    public double Predict(double x1, double x2) => Forward(x1, x2).Prediction;

    public ForwardResult Inspect(double x1, double x2) => Forward(x1, x2);

    public EvaluationResult Evaluate(IReadOnlyList<Sample> examples)
    {
        if (examples.Count == 0)
        {
            throw new InvalidOperationException("XOR evaluation needs at least one example.");
        }

        var loss = 0.0;
        var gradients = new double[_parameters.Length];
        foreach (var example in examples)
        {
            if (!double.IsFinite(example.Target))
            {
                throw new InvalidOperationException("XOR targets must be finite.");
            }

            var (prediction, hidden) = Forward(example.X1, example.X2);
            var error = prediction - example.Target;
            loss += error * error / examples.Count;
            var outputGradient = 2 * error / examples.Count;
            gradients[16] += outputGradient;
            for (var unit = 0; unit < HiddenUnits; unit++)
            {
                gradients[12 + unit] += outputGradient * hidden[unit];
                var hiddenGradient = outputGradient * _parameters[12 + unit] * (1 - hidden[unit] * hidden[unit]);
                var offset = 3 * unit;
                gradients[offset] += hiddenGradient * example.X1;
                gradients[offset + 1] += hiddenGradient * example.X2;
                gradients[offset + 2] += hiddenGradient;
            }
        }
        if (!double.IsFinite(loss) || gradients.Any(value => !double.IsFinite(value)))
        {
            throw new InvalidOperationException("XOR loss or gradient overflowed.");
        }

        return new EvaluationResult(loss, gradients);
    }

    public void TrainStep(IReadOnlyList<Sample> examples, double learningRate)
    {
        if (!double.IsFinite(learningRate) || learningRate <= 0)
        {
            throw new InvalidOperationException("The XOR learning rate must be positive and finite.");
        }

        var gradients = Evaluate(examples).Gradients;
        var updated = new double[_parameters.Length];
        for (var i = 0; i < updated.Length; i++)
        {
            updated[i] = _parameters[i] - learningRate * gradients[i];
        }

        if (updated.Any(value => !double.IsFinite(value)))
        {
            throw new InvalidOperationException("XOR update overflowed. Use a smaller learning rate.");
        }

        Array.Copy(updated, _parameters, updated.Length);
    }

    public double[] NumericalGradients(IReadOnlyList<Sample> examples, double epsilon = 1e-6)
    {
        if (!double.IsFinite(epsilon) || epsilon <= 0)
        {
            throw new InvalidOperationException("The finite-difference epsilon must be positive and finite.");
        }

        var gradients = new double[_parameters.Length];
        for (var i = 0; i < gradients.Length; i++)
        {
            var plus = new XorNetwork(this);
            var minus = new XorNetwork(this);
            plus._parameters[i] += epsilon;
            minus._parameters[i] -= epsilon;
            gradients[i] = (plus.Evaluate(examples).Loss - minus.Evaluate(examples).Loss) / (2 * epsilon);
        }
        return gradients;
    }
}
