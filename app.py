import streamlit as st
import openai
import librosa
import matplotlib.pyplot as plt
import os

# Set up OpenAI Client
openai.api_key = st.secrets["openai"]["openai_api_key"]
openai_model = 'davinci'

# Set up OpenAI Playground Trained Model
playground_model = "ezra"

# Set up audio processing parameters
sr = 44100  # Sample rate
n_fft = 2048  # FFT size
hop_length = n_fft // 4  # Hop length for STFT

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

# Generate equalization curve based on the uploaded audio file and additional prompt
if audio_file is not None:
    # Load the audio file and extract its spectral content
    try:
        y, _ = librosa.load(audio_file, sr=sr)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
        S = librosa.amplitude_to_db(abs(D), ref=1.0)
    except:
        st.error("Unable to load audio file. Please try again with a different file.")
        st.stop()

    # Use the OpenAI API to generate a text prompt for the equalization curve
    prompt = f"Generate an equalization curve for the uploaded audio file. The audio has {additional_prompt}. The desired equalization should enhance the {specific_quality} and reduce the {specific_issue}."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=4000
    )

    # Add error handling for empty response
    if not response.choices:
        st.error("No equalization curve found. Please try again with a different prompt.")
        st.stop()

    eq_text = response.choices[0].text

    # Validate and parse the equalization curve from the generated text
    eq_curve = []
    try:
        eq_curve = list(map(float, eq_text.strip().split()))
    except ValueError:
        st.error("Invalid equalization curve format. Please try again with a different prompt.")
        st.stop()

    # Check if the parsed equalization curve is empty
    if not eq_curve:
        st.error("Equalization curve not found. Please try again with a different prompt.")
        st.stop()

    # Plot the equalization curve on the Matplotlib figure
    ax.plot(eq_curve)

    # Set the labels and title for the Matplotlib figure
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("Recommended Equalization Curve")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)
