import sounddevice as sd

fs = 16000
duration = 2  # seconds
sd.default.samplerate = fs
sd.default.channels = 1
myrecording = sd.rec(int(duration * fs), dtype='float64')
sd.wait()

from scipy.io.wavfile import write
filename = "output_audio.wav"
write(filename, fs, myrecording)
print(f"Audio saved to {filename}")

from transformers import WhisperForConditionalGeneration, WhisperProcessor
model_name = "openai/whisper-tiny"

from transformers import WhisperForConditionalGeneration, WhisperProcessor
model_name = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

inputs = processor(myrecording, sampling_rate=fs, return_tensors="pt")