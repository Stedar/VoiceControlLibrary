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
        List<string> PeriodsTokens;

        PredictionEngine<CommandsData, IssuePrediction> _predEngine;
        PredictionEngine<CommandsData, IssuePrediction> _predEngine_periods;

        public void BuildModel()
        {

            //основная модель - поиск команд
            MLContext _mlContext = new MLContext();
            String FilePath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            IDataView trainingDataView = _mlContext.Data.LoadFromTextFile<CommandsData>(FilePath + "\\train_commands.csv", separatorChar: ';', hasHeader: false);

            // Create an IEnumerable from IDataView
            IEnumerable<CommandsData> trainingDataViewEnumerable = _mlContext.Data.CreateEnumerable<CommandsData>(trainingDataView, reuseRowObject: true);
            var r_text_list = trainingDataViewEnumerable.Select(r => r.TextR).ToList();

            //создаем словарь (токены) из слов для команд

            char[] separators = new char[] { ' ', '.' };
            string string_words = string.Join(" ", r_text_list.ToArray());
            string[] test_subs = string_words.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            List<string> words_command_list = new List<string>(test_subs);

            CommandsTokens = words_command_list.Distinct().ToList();
            //сохраняем список токенов в файл
            System.IO.File.WriteAllLines(FilePath + "\\CommandsTokens.txt", CommandsTokens);

            //заполним также список всех команд (label) в отдельный список. 
            CommandsList = trainingDataViewEnumerable.Select(r => r.Command).ToList().Distinct().ToList();
            
            //сохраняем список команд в файл
            System.IO.File.WriteAllLines(FilePath + "\\Commands.txt", CommandsList);

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Command", outputColumnName: "Label")
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "TextR", outputColumnName: "Features"))
            .AppendCacheCheckpoint(_mlContext);

            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            ITransformer _trainedModel = trainingPipeline.Fit(trainingDataView);

             _predEngine = _mlContext.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel);
            // Save model
            _mlContext.Model.Save(_trainedModel, trainingDataView.Schema, FilePath + "\\model_main_cls.zip");


            //вспомогательная модель - отбор по периодам
            MLContext _mlContext_periods = new MLContext();
            trainingDataView = _mlContext.Data.LoadFromTextFile<CommandsData>(FilePath + "\\train_periods.csv", separatorChar: ';', hasHeader: false);

            // Create an IEnumerable from IDataView
            trainingDataViewEnumerable = _mlContext_periods.Data.CreateEnumerable<CommandsData>(trainingDataView, reuseRowObject: true);
            r_text_list = trainingDataViewEnumerable.Select(r => r.TextR).ToList();

            string_words = string.Join(" ", r_text_list.ToArray());
            test_subs = string_words.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            words_command_list = new List<string>(test_subs);

            PeriodsTokens = words_command_list.Distinct().ToList();
            //сохраняем токены в файл
            System.IO.File.WriteAllLines(FilePath + "\\PeriodsTokens.txt", PeriodsTokens);


            var pipeline_periods = _mlContext_periods.Transforms.Conversion.MapValueToKey(inputColumnName: "Command", outputColumnName: "Label")
            .Append(_mlContext_periods.Transforms.Text.FeaturizeText(inputColumnName: "TextR", outputColumnName: "Features"))
            .AppendCacheCheckpoint(_mlContext_periods);

            var trainingPipelinePeriods = pipeline_periods.Append(_mlContext_periods.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext_periods.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            ITransformer _trainedModel_periods = trainingPipelinePeriods.Fit(trainingDataView);

            _predEngine_periods = _mlContext_periods.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel_periods);

            // Save model
            _mlContext.Model.Save(_trainedModel_periods, trainingDataView.Schema, FilePath + "\\model_periods_cls.zip");


        }

        public void LoadModels()
        {
            if (CommandsList!=null)
                CommandsList.Clear();
            CommandsList = new List<string>();
            if (CommandsTokens != null)
                CommandsTokens.Clear();
            CommandsTokens = new List<string>();
            if (PeriodsTokens != null)
                PeriodsTokens.Clear();
            PeriodsTokens = new List<string>();

            //Define DataViewSchema for data preparation pipeline and trained model
            DataViewSchema modelSchema;
            MLContext _mlContext = new MLContext();
            String FilePath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            // Load trained model
            ITransformer _trainedModel = _mlContext.Model.Load(FilePath + "\\model_main_cls.zip", out modelSchema);

            var logFile = File.ReadAllLines(FilePath + "\\Commands.txt");
            foreach (var s in logFile) CommandsList.Add(s);

            logFile = File.ReadAllLines(FilePath + "\\CommandsTokens.txt");
            foreach (var s in logFile) CommandsTokens.Add(s);

       
            //_predEngine_periods = _mlContext_periods.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel_periods);
            _predEngine = _mlContext.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel);

            MLContext _mlContext_periods = new MLContext();

            // Load trained model
            ITransformer _trainedModel_periods = _mlContext_periods.Model.Load(FilePath + "\\model_periods_cls.zip", out modelSchema);


            logFile = File.ReadAllLines(FilePath + "\\PeriodsTokens.txt");
            foreach (var s in logFile) PeriodsTokens.Add(s);

            _predEngine_periods = _mlContext_periods.Model.CreatePredictionEngine<CommandsData, IssuePrediction>(_trainedModel_periods);

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

        public string PredictPeriodSyntax(string RecognizedString)
        {
            string result = "None";

            if (_predEngine_periods == null)
                return "_predEngine not found";

            char[] separators = new char[] { ' ', '.' };
            string t = RecognizedString.ToLower();
            t = t.Replace(",", "").Replace("ё", "е");

            string[] subs = t.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            List<string> list_test = new List<string>(subs);

            IEnumerable<string> CommonList = list_test.Intersect(PeriodsTokens);
            var singleString = string.Join(" ", CommonList.ToArray());

            if (singleString.Length == 0)
                singleString = "непонтяно";

            CommandsData issue = new CommandsData()
            {
                TextR = singleString,
            };


            var prediction = _predEngine_periods.Predict(issue);
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
