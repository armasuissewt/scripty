import time
import torch
import numpy as np
from faster_whisper import WhisperModel


def check_environment():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        print(f"CUDA is available. Version: {cuda_version}")
    else:
        print("CUDA is not available.")

    # Print the versions of the main libraries
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("Using faster-whisper for transcription.")


def scripty():
    # Load the faster-whisper model
    print("Before loading model")
    model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
    print("After loading model")

    # Check if the model is loaded
    if model is not None:
        print("Model loaded successfully!")
    else:
        print("Model failed to load.")

    # Load and process the audio file
    segments, info = model.transcribe("samples/german.mp3")

    # Decode the audio and measure the time taken
    start_time = time.time()

    # Collect the recognized text
    recognized_text = ""
    for segment in segments:
        recognized_text += segment.text

    end_time = time.time()
    print(f"Decoding took {end_time - start_time:.2f} seconds")

    # Print the recognized text
    print("Text: " + recognized_text)


if __name__ == "__main__":
    check_environment()
    scripty()
