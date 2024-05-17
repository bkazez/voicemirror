import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from scipy.signal import savgol_filter

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048  # Increased chunk size

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# German vowel formant frequencies (F1, F2) in Hz
german_vowels = {
    'iː': (300, 2250),   # long 'i'
    'ɪ': (400, 1900),    # short 'i'
    'eː': (400, 2000),   # long 'e'
    'ɛ': (600, 1700),    # short 'e'
    'aː': (750, 1200),   # long 'a'
    'ɑ': (850, 1250),    # short 'a'
    'oː': (400, 800),    # long 'o'
    'ɔ': (500, 900),     # short 'o'
    'uː': (350, 700),    # long 'u'
    'ʊ': (450, 900),     # short 'u'
    'øː': (400, 1600),   # long 'ö'
    'œ': (600, 1500),    # short 'ö'
    'yː': (300, 1700),   # long 'ü'
    'ʏ': (450, 1600),    # short 'ü'
}

# Plot setup
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim(0, 1000)  # Adjusted range for F1
ax.set_ylim(0, 2500)  # Adjusted range for F2
ax.set_xlabel('F1 (Hz)')
ax.set_ylabel('F2 (Hz)')

# Add German vowel reference points
for vowel, (f1, f2) in german_vowels.items():
    ax.scatter(f1, f2, c='red')
    ax.text(f1, f2, vowel, fontsize=12, ha='right', color='red')

# Smoothing parameters
formant_buffer = []

def get_formants(audio_sample, rate):
    sound = parselmouth.Sound(audio_sample, sampling_frequency=rate)
    formant = sound.to_formant_burg()
    formants = []
    for i in range(1, 3):  # Extract first two formants
        formants.append(formant.get_value_at_time(i, sound.duration / 2))
    return formants

while True:
    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        formants = get_formants(data, RATE)
        
        if len(formants) >= 2:
            F1, F2 = formants[:2]
            formant_buffer.append((F1, F2))
            
            # Apply smoothing
            if len(formant_buffer) > 10:
                formant_buffer = formant_buffer[-10:]
                F1_smooth = savgol_filter([f[0] for f in formant_buffer], window_length=7, polyorder=2)
                F2_smooth = savgol_filter([f[1] for f in formant_buffer], window_length=7, polyorder=2)
                sc.set_offsets(np.c_[F1_smooth[-1], F2_smooth[-1]])
            else:
                sc.set_offsets(np.c_[F1, F2])
        
        plt.pause(0.01)
    except IOError:
        pass

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
