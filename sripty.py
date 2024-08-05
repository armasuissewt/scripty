import time
import whisper
import torch
import numpy as np


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
    print(f"Whisper version: {whisper.__version__}")


def scripty():
    # Load the Whisper model and force it to use the CPU
    model = whisper.load_model("base")

    # Load audio file and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("samples/german.mp3")
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio
    start_time = time.time()
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    end_time = time.time()
    print(f"Decoding took {end_time - start_time:.2f} seconds")

    # Print the recognized text
    print("Text: " + result.text)


if __name__ == "__main__":
    check_environment()
    scripty()
