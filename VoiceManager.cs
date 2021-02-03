using System;
using System.Runtime.InteropServices;
using NAudio.Wave;
using System.IO;
using Google.Cloud.Speech.V1;
using Ml_command_module;
using System.Collections.Generic;

namespace VoiceControlLibrary
{
    [ComVisible(true)]
    [Guid("4AF54029-0EFB-44EE-8990-F0B7BC757314")]
    public class VoiceManager : IVoiceManager
    {

        private WaveInEvent waveIn;
        private String outputFilePath;
        private WaveFileWriter writer;
        private float max_v; //for volume
        private bool isWriting;
        private float TimeDelay;
        private DateTime LastWritingDateTime;
        private SpeechClient speech;
        private RecognitionConfig config;
        private String strRecgnResult;
        private float IdleTimeAmount;
        private bool isResultRecieved;
        private bool isRecordStarted;

        private ML_text_analizer mlTextanalyzer;

        public VoiceManager()
        {
            var outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "NAudio");
            Directory.CreateDirectory(outputFolder);
            outputFilePath = Path.Combine(outputFolder, "recorded.wav");
            waveIn = new WaveInEvent();
            waveIn.WaveFormat = new WaveFormat(44100, 16, 1);
            waveIn.DataAvailable += waveIn_DataAvailable;
            waveIn.RecordingStopped += waveIn_RecordingStopped;

            writer = null;
            max_v = 0;
            isWriting = false;
            TimeDelay = 1; //seconds
            strRecgnResult = "";
            IdleTimeAmount = 10; //seconds
            isResultRecieved = false;
            isRecordStarted = false;

            speech = SpeechClient.Create();
            config = new RecognitionConfig
            {
                LanguageCode = LanguageCodes.Russian.Russia
            };

            mlTextanalyzer = new ML_text_analizer();
        }

        public string TestS()
        {
            return System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        }
        public void InitVoiceManager()
        {
            mlTextanalyzer.BuildModel();

        }

        public string[] GetCommandsList()
        {
            if (mlTextanalyzer == null)
                return null;
            else
                return mlTextanalyzer.GetCommandsList().ToArray();
        }
        ~VoiceManager()
        {
            DisableSystem();
        }

        public bool IsRecordStarted()
        {
            return isRecordStarted;
        }


        public void DisableSystem()
        {
            isWriting = false;
            waveIn.StopRecording();

        }

        private void StopRecord()
        {
            waveIn.StopRecording();

        }

        public void StartRecord()
        {
            if (isRecordStarted)
                return;

            writer = new WaveFileWriter(outputFilePath, waveIn.WaveFormat);
            isWriting = false;
            waveIn.StartRecording();
            LastWritingDateTime = DateTime.Now;
            strRecgnResult = "";
            max_v = 0.0f;
            isRecordStarted = true;
        }

        public string GetRecognitionResult()
        {
            return strRecgnResult;
        }


        public void SetCommandRecognizedString(string sValue)
        {
            strRecgnResult = sValue;
        }
        public string GetCommandRecognizedString()
        {
          // return "test";
            if (strRecgnResult == "")
                return "нераспознано";
            else
                return mlTextanalyzer.PredictCommand(strRecgnResult);

        }
        public string GetPeriodRecognizedString()
        {
            // return "test";
            if (strRecgnResult == "")
                return "нераспознано";
            else
                return mlTextanalyzer.PredictPeriodSyntax(strRecgnResult);

        }
        public void ClearRecognitionResult()
        {
            strRecgnResult = "";
        }

        public float GetRecordVolume()
        {
            return max_v;
        }

        private void waveIn_RecordingStopped(object sender, EventArgs e)

        {
            waveIn.Dispose();
            writer.Close();
            writer = null;
            isRecordStarted = false;
            if (this.isWriting)
                SpeechToText();
        }


        private void waveIn_DataAvailable(object sender, WaveInEventArgs e)
        {
            TimeSpan ts = TimeSpan.Zero;
            max_v = 0;
            // interpret as 16 bit audio
            for (int index = 0; index < e.BytesRecorded; index += 2)
            {
                short sample = (short)((e.Buffer[index + 1] << 8) |
                                        e.Buffer[index + 0]);
                // to floating point
                var sample32 = sample / 32768f;
                // absolute value 
                if (sample32 < 0) sample32 = -sample32;
                // is this the max value?
                if (sample32 > max_v) max_v = sample32;
            }

            ts = DateTime.Now - LastWritingDateTime;

            if (max_v > 0.1 || (isWriting & ts.Seconds < TimeDelay))
            {
                writer.Write(e.Buffer, 0, e.BytesRecorded);
                if (max_v > 0.1)
                    LastWritingDateTime = DateTime.Now;
                if (!isWriting)
                    isWriting = true;
            }
            else if (isWriting & (max_v <= 0.1 & ts.Seconds >= TimeDelay))
            {
                StopRecord();
            }
            else if (!isWriting & ts.Seconds >= IdleTimeAmount)// если молчание длиится долго, перезапускаем Запись
            {
                StopRecord();

            }

        }
        private void SpeechToText()
        {
            strRecgnResult = "";
            var audio = RecognitionAudio.FromFile(outputFilePath);

            var response = speech.Recognize(config, audio);

            foreach (var result in response.Results)
            {
                foreach (var alternative in result.Alternatives)
                {
                    strRecgnResult += ((alternative.Transcript) + "...");
                }
            }
            isResultRecieved = true;
        }

    }
}
