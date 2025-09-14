import threading
import queue
import time
import torch
import numpy as np
import sounddevice as sd
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Config
fs = 16000
block_duration = 2.0  # seconds
block_size = int(fs * block_duration)
channels = 1
# model_name = "openai/whisper-large-v2"
model_name = "openai/whisper-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

audio_queue = queue.Queue()
stop_event = threading.Event()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_chunk = indata.copy().squeeze()
    if audio_chunk.shape[0] == block_size:
        audio_queue.put(audio_chunk)


def transcribe_audio(model, processor):
    print("Inference thread started.")

    prev_chunk = None
    transcriptions = []

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Process single chunk
        inputs = processor(audio_chunk, sampling_rate=fs, return_tensors="pt").input_features
        predicted_ids = model.generate(inputs.to(device))
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        print(f"[Single] {transcription}")
        transcriptions.append(transcription)

        # Stitch with previous if available
        if prev_chunk is not None:
            stitched = np.append(prev_chunk, audio_chunk)
            inputs = processor(stitched, sampling_rate=fs, return_tensors="pt").input_features
            predicted_ids = model.generate(inputs.to(device))
            stitched_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            print(f"[Stitched] {stitched_transcription}")
            transcriptions[-1] = stitched_transcription

        prev_chunk = audio_chunk

    print("Final transcriptions:", transcriptions)


if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    print("Starting real-time transcription...")

    # Start transcriber thread
    transcriber = threading.Thread(target=transcribe_audio, kwargs={"model": model, "processor": processor})
    transcriber.start()

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=channels,
            dtype='float32',
            callback=audio_callback,
            blocksize=block_size
        ):
            print("Recording... Press Ctrl+C to stop.")
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stop_event.set()
        transcriber.join()
        print("Stopped.")
