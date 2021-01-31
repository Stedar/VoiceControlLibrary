using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;


namespace Ml_command_module
{

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Command;
        public float Probability { get; set; }
    }

    public class CommandsData
    {
        [LoadColumn(0)]
        [ColumnName("TextR")]
        public string TextR { get; set; }

        [LoadColumn(1)]
        [ColumnName("Command")]
        public string Command { get; set; }

    }


    class ML_text_analizer
    {
        List<string> CommandsList;
        List<string>CommandsTokens;
        PredictionEngine<CommandsData, IssuePrediction> _predEngine;
        public void BuildModel()
        {

            MLContext _mlContext = new MLContext();
            String FilePath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + "\\train_commands.csv";
            IDataView trainingDataView = _mlContext.Data.LoadFromTextFile<CommandsData>(FilePath, separatorChar: ';', hasHeader: false);

            // Create an IEnumerable from IDataView
            IEnumerable<CommandsData> trainingDataViewEnumerable = _mlContext.Data.CreateEnumerable<CommandsData>(trainingDataView, reuseRowObject: true);
            var r_text_list = trainingDataViewEnumerable.Select(r => r.TextR).ToList();

            //создаем словарь (токены) из слов для команд

            char[] separators = new char[] { ' ', '.' };
            string string_words = string.Join(" ", r_text_list.ToArray());
            string[] test_subs = string_words.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            List<string> words_command_list = new List<string>(test_subs);

            CommandsTokens = words_command_list.Distinct().ToList();

            //заполним также список всех команд (label) в отдельный список. 
            CommandsList = trainingDataViewEnumerable.Select(r => r.Command).ToList().Distinct().ToList();


            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Command", outputColumnName: "Label")
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "TextR", outputColumnName: "Features"))
            .AppendCacheCheckpoint(_mlContext);

            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            ITransformer _trainedModel = trainingPipeline.Fit(trainingDataView);

             _predEngine = _mlContext.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel);


        }
        
        public string PredictCommand(string RecognizedString)
        {
            string result = "None";

            if (_predEngine == null)
                return "_predEngine not found";

            char[] separators = new char[] { ' ', '.' };
            string t = RecognizedString.ToLower();
            t = t.Replace(",", "").Replace("ё", "е");

            string[] subs = t.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            List<string> list_test = new List<string>(subs);

            IEnumerable<string> CommonList = list_test.Intersect(CommandsTokens);
            var singleString = string.Join(" ", CommonList.ToArray());

            if (singleString.Length == 0)
                singleString = "непонтяно";

            CommandsData issue = new CommandsData()
            {
                TextR = singleString,
            };


            var prediction = _predEngine.Predict(issue);
            result = prediction.Command;
           // Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Command} ===============");
            
            return result;
        }
   
        public List<string> GetCommandsList()
        {
            return CommandsList;
        }
    }




}
