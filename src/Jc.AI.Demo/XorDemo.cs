using Jc.AI.Models;

namespace Jc.AI.Demo;

internal static class XorDemo
{
    private static IReadOnlyList<XorNetwork.Sample> Examples { get; } =
        Array.AsReadOnly(new XorNetwork.Sample[]
        {
            new(0, 0, 0),
            new(0, 1, 1),
            new(1, 0, 1),
            new(1, 1, 0)
        });

    private static (double Weight1, double Weight2, double Bias, double Loss) TrainLinearBaseline()
    {
        var weight1 = 0.0;
        var weight2 = 0.0;
        var bias = 0.0;
        for (var step = 0; step < 10000; step++)
        {
            var gradient1 = 0.0;
            var gradient2 = 0.0;
            var biasGradient = 0.0;
            foreach (var (x1, x2, target) in Examples)
            {
                var gradient = 2 * (weight1 * x1 + weight2 * x2 + bias - target) / Examples.Count;
                gradient1 += gradient * x1;
                gradient2 += gradient * x2;
                biasGradient += gradient;
            }

            weight1 -= 0.1 * gradient1;
            weight2 -= 0.1 * gradient2;
            bias -= 0.1 * biasGradient;
        }

        var loss = Examples.Average(example =>
            Math.Pow(weight1 * example.X1 + weight2 * example.X2 + bias - example.Target, 2));
        return (weight1, weight2, bias, loss);
    }

    public static void Run(int seed)
    {
        var baseline = TrainLinearBaseline();
        var network = new XorNetwork(seed);
        var analytic = network.Evaluate(Examples).Gradients;
        var numerical = network.NumericalGradients(Examples);
        if (analytic.Where((value, i) => !double.IsFinite(numerical[i]) || Math.Abs(value - numerical[i]) > 1e-6).Any())
        {
            throw new InvalidOperationException("The XOR gradient check failed.");
        }

        Console.WriteLine("Phase 5: XOR with a 2-4-1 network (tanh hidden units, linear output).");
        Console.WriteLine($"Seed: {seed} | learning rate: 0.1 | full-batch updates: 10000");
        Console.WriteLine(FormattableString.Invariant($"Trained linear baseline MSE: {baseline.Loss:F6}"));
        Console.WriteLine("Gradient check (17 parameters, epsilon=1E-6, absolute tolerance=1E-6)");
        Console.WriteLine("Step | Network mean squared error");
        for (var step = 0; step <= 10000; step++)
        {
            if (step is 0 or 1 or 100 or 1000 or 10000)
            {
                Console.WriteLine(FormattableString.Invariant($"{step} | {network.Evaluate(Examples).Loss:E6}"));
            }

            if (step < 10000)
            {
                network.TrainStep(Examples, 0.1);
            }
        }

        Console.WriteLine("x1 x2 | Target | Linear | Network | Network class");
        var correct = 0;
        foreach (var (x1, x2, target) in Examples)
        {
            var linear = baseline.Weight1 * x1 + baseline.Weight2 * x2 + baseline.Bias;
            var prediction = network.Predict(x1, x2);
            var classification = prediction >= 0.5 ? 1 : 0;
            if (classification == target)
            {
                correct++;
            }

            Console.WriteLine(
                FormattableString.Invariant(
                    $"{x1:G} {x2:G} | {target:G} | {linear:F6} | {prediction:F6} | {classification}"));
        }

        Console.WriteLine($"Network accuracy: {correct}/4 (threshold 0.5)");
        Console.WriteLine("Network outputs are scores, not probabilities. No text datasets are used.");
    }
}