import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load the audio file
audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path)

# Plot the waveform
plt.figure(figsize=(14, 5))
plt.plot(y)
plt.title('Waveform of the Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()