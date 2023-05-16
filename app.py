from typing import Tuple
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
import openai

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Constants
sr = 22050
n_fft = 2048
hop_length = 512

# Set up Matplotlib figure for the equalization curve
fig, ax = plt.subplots()

# App framework
st.title("Ezra🎚️🏳️‍⚧️")

# Get user input for additional text prompt
additional_prompt = st.text_input("Describe the qualities of your desired EQ")

# Get user input for specific quality and issue
specific_quality = st.text_input("Specify the specific quality you want to enhance")
specific_issue = st.text_input("Specify the specific issue you want to reduce")

# Upload an audio file
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "ogg"])


# Function to extract audio features
def extract_audio_features(audio_file: str) -> Tuple[np.ndarray, int]:
    """Loads an audio file and returns its spectral features and sampling rate"""
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    except FileNotFoundError as err:
        st.error("Audio file not found")
    except Exception as err:
        st.error("Error processing audio file")
    return D, sr


# Generate equalization curve
def generate_eq_curve(D: np.ndarray, additional_prompt: str,
                      specific_quality: str, specific_issue: str) -> np.ndarray:
    # TODO: Implement equalization curve generation based on audio features and user input
    # For demonstration purposes, let's generate a random equalization curve
    num_bins = D.shape[0]
    eq_curve = np.random.uniform(low=-12, high=12, size=num_bins)
    return eq_curve


# Plot the equalization curve
def plot_eq_curve(curve: np.ndarray, sr: int, ax: plt.Axes) -> None:
    # TODO: Implement equalization curve plotting
    # For demonstration purposes, let's plot the equalization curve as a line
    frequency_bins = np.arange(curve.shape[0]) * (sr / 2) / curve.shape[0]
    ax.plot(frequency_bins, curve)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("Equalization Curve")


# Generate equalization curve based on the uploaded audio file and additional prompt
if audio_file is not None:
    # Extract audio features
    D, sr = extract_audio_features(audio_file)

    # Generate EQ curve
    curve = generate_eq_curve(D, additional_prompt, specific_quality, specific_issue)

    # Plot the equalization curve
    plot_eq_curve(curve, sr, ax)

    # Display the equalization curve
    st.pyplot(fig)
