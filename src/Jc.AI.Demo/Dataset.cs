namespace Jc.AI.Demo;

internal static class Dataset
{
    public record DatasetMeasurement(int Lines, int Words, int CodeUnits);
    
    public static string Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "data", "facts.txt");
        var text = File.ReadAllText(path);
        return string.IsNullOrWhiteSpace(text) ? throw new InvalidOperationException("The bundled dataset is empty.") : text;
    }

    public static DatasetMeasurement Measure(string text)
    {
        var lines = 0;
        using var reader = new StringReader(text);
        while (reader.ReadLine() is not null)
        {
            lines++;
        }

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
        return new DatasetMeasurement(lines, words, text.Length);
    }
}