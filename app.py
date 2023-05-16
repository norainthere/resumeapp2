from typing import Tuple
import openai
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np

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
    # Convert user input to prompt format
    prompt = f"Additional Prompt: {additional_prompt}\nSpecific Quality: {specific_quality}\nSpecific Issue: {specific_issue}\n"
    
    # Generate equalization curve using OpenAI language model
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    # Extract the generated curve from the OpenAI response
    generated_curve = response.choices[0].text.strip()
    
    # Parse the generated curve into an array of floats
    eq_curve = np.array(list(map(float, generated_curve.split())))
    
    return eq_curve


# Plot the equalization curve
def plot_eq_curve(curve: np.ndarray, sr: int, ax: plt.Axes) -> None:
    # Plot the equalization curve
    frequency_bins = np.arange(curve.shape[0]) * (sr / 2) / curve.shape[0]
    ax.plot(frequency_bins, curve)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("Equalization Curve")


# Check if the OpenAI API key is available in Streamlit secrets
if "openai_api_key" not in st.secrets:
    st.error("OpenAI API key not found in Streamlit secrets.")
    st.stop()

# Set the OpenAI API key from Streamlit secrets
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key


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

