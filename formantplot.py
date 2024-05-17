import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from scipy.signal import savgol_filter
import json
import argparse

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048  # Increased chunk size

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Argument parser for CLI
parser = argparse.ArgumentParser(description='Real-time formant plotter with vowel sets.')
parser.add_argument('--vowel_set', type=str, required=True, help='The full path to the vowel set JSON file (e.g., "/path/to/de.json").')
args = parser.parse_args()

# Function to load vowel sets
def load_vowel_set(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load and display the initial vowel set
current_vowel_set = load_vowel_set(args.vowel_set)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim(0, 1000)  # Adjusted range for F1
ax.set_ylim(0, 2500)  # Adjusted range for F2
ax.set_xlabel('F1 (Hz)')
ax.set_ylabel('F2 (Hz)')

# Add vowel reference points
for vowel, (f1, f2) in current_vowel_set.items():
    ax.scatter(f1, f2, c='red')
    ax.text(f1, f2, vowel, fontsize=12, ha='right', color='red')

# Smoothing parameters
formant_buffer = []
harmonic_buffer = []
harmonic_lines = []

def get_formants(audio_sample, rate):
    sound = parselmouth.Sound(audio_sample, sampling_frequency=rate)
    formant = sound.to_formant_burg()
    formants = []
    for i in range(1, 3):  # Extract first two formants
        formants.append(formant.get_value_at_time(i, sound.duration / 2))
    return formants

def get_harmonics(audio_sample, rate):
    n = len(audio_sample)
    k = np.arange(n)
    T = n / rate
    frq = k / T
    frq = frq[range(n // 2)]

    Y = np.fft.fft(audio_sample) / n
    Y = Y[range(n // 2)]
    Y = np.abs(Y)

    peaks = np.argsort(Y)[-2:][::-1]
    harmonics = frq[peaks]

    # Ensure H2 is approximately twice H1
    if harmonics[1] > harmonics[0] * 1.8 and harmonics[1] < harmonics[0] * 2.2:
        return harmonics[:2]
    else:
        return harmonics[0], harmonics[0] * 2

# Run the real-time formant plot
def run_real_time_plot():
    global formant_buffer, harmonic_buffer, harmonic_lines
    while True:
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            formants = get_formants(data, RATE)
            harmonics = get_harmonics(data, RATE)
            
            if len(formants) >= 2:
                F1, F2 = formants[:2]
                formant_buffer.append((F1, F2))
                
                # Apply smoothing to formants
                if len(formant_buffer) > 30:
                    formant_buffer = formant_buffer[-30:]
                    F1_smooth = savgol_filter([f[0] for f in formant_buffer], window_length=21, polyorder=2)[-1]
                    F2_smooth = savgol_filter([f[1] for f in formant_buffer], window_length=21, polyorder=2)[-1]
                    sc.set_offsets(np.c_[F1_smooth, F2_smooth])
                else:
                    sc.set_offsets(np.c_[F1, F2])
            
            if len(harmonics) >= 2:
                H1, H2 = harmonics[:2]
                harmonic_buffer.append((H1, H2))
                
                # Apply smoothing to harmonics
                if len(harmonic_buffer) > 30:
                    harmonic_buffer = harmonic_buffer[-30:]
                    H1_smooth = savgol_filter([h[0] for h in harmonic_buffer], window_length=21, polyorder=2)[-1]
                    H2_smooth = savgol_filter([h[1] for h in harmonic_buffer], window_length=21, polyorder=2)[-1]
                else:
                    H1_smooth, H2_smooth = H1, H2
                
                # Clear old harmonics
                while harmonic_lines:
                    line = harmonic_lines.pop(0)
                    line.remove()
                    del line

                # Plot H1 and H2
                line1 = ax.axvline(x=H1_smooth, color='blue', linestyle='--')
                line2 = ax.axvline(x=H2_smooth, color='green', linestyle='--')
                text1 = ax.text(H1_smooth, 2400, 'H1', fontsize=8, ha='center', color='blue')
                text2 = ax.text(H2_smooth, 2400, 'H2', fontsize=8, ha='center', color='green')
                harmonic_lines.extend([line1, line2, text1, text2])
            
            plt.pause(0.01)
        except IOError:
            pass

# Start the real-time plot
run_real_time_plot()

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
