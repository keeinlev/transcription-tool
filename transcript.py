import torch
import numpy as np
import sounddevice as sd
from transformers import WhisperForConditionalGeneration, WhisperProcessor



if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    transcriptions = []

    fs = 16000
    sd.default.samplerate = fs
    sd.default.channels = 1
    rec_duration = 2  # seconds
    prev_recording = None

    print("Recording started.")
    for _ in range(10):
        recording = sd.rec(int(rec_duration * fs), dtype='float64').squeeze()
        sd.wait()

        inputs = processor(recording, sampling_rate=fs, return_tensors="pt").input_features
        predicted_ids = model.generate(inputs.to(device))
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        transcriptions.append(transcription)
        print(transcriptions)

        if prev_recording is not None:
            stitched = np.append(prev_recording, recording)
            inputs = processor(stitched, sampling_rate=fs, return_tensors="pt").input_features
            predicted_ids = model.generate(inputs.to(device))
            stitched_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            transcriptions = transcriptions[:-1] + [stitched_transcription]
            print(transcriptions)
            # print(t1, t2)
        else:
            print(transcription)

        prev_recording = recording

    # print(transcriptions)