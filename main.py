#!/usr/bin/env python
"""
Audio Analysis Script

This script performs comprehensive audio analysis on MP3 files including:
- Waveform visualization
- Channel separation and comparison
- Signal histogram analysis
- Frequency spectrum analysis using FFT
- Signal normalization and quantization

Requirements:
- pydub
- matplotlib
- numpy

Usage:
    python main.py
"""

from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# Create output directory if it doesn't exist
output_dir = "audio_plots"
os.makedirs(output_dir, exist_ok=True)

def load_audio(file_path):
    """Load audio file and extract basic information"""
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    channels = audio.channels
    return audio, samples, sample_rate, channels

def plot_channels_comparison(left_channel, right_channel, sample_rate):
    """Create visualization comparing left and right audio channels"""
    time = np.linspace(0, len(left_channel) / sample_rate, len(left_channel))

    plt.figure(figsize=(20, 8))
    plt.title("Ляв и десен канал")
    plt.xlabel("Продължителност (s)")
    plt.ylabel("Амплитуда")
    plt.plot(time, left_channel, label="Ляв канал", alpha=0.7)
    plt.plot(time, right_channel, label="Десен канал", alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "channels.png"))
    plt.close()

def plot_waveform(samples, sample_rate, channels):
    """Create waveform visualization of the entire audio signal"""
    duration = len(samples) / (float(sample_rate) * channels)
    time = np.linspace(0, duration, len(samples))

    plt.figure(figsize=(20, 8))
    plt.title("Визуализация на съдържанието на файл: my_voice_data.mp3")
    plt.xlabel("Продължителност (s)")
    plt.ylabel("Амплитуда")
    plt.plot(time, samples)
    plt.savefig(os.path.join(output_dir, "waveform.png"))
    plt.close()

def plot_signal_histogram(samples):
    """Create histogram of the raw audio signal values"""
    plt.figure(figsize=(20, 8))
    plt.title("Хистограма на сигнала")
    plt.xlabel("Стойности на сигнала")
    plt.ylabel("Брой срещания")
    plt.hist(samples, bins=1000, color='b', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "signal_histogram.png"))
    plt.close()

def plot_frequency_spectrum(left_channel, right_channel, sample_rate):
    """Create frequency spectrum analysis using FFT for both channels"""
    # FFT for left channel
    left_channel_fft = np.fft.fft(left_channel)
    left_channel_fft = left_channel_fft[:len(left_channel_fft) // 2]

    # FFT for right channel
    right_channel_fft = np.fft.fft(right_channel)
    right_channel_fft = right_channel_fft[:len(right_channel_fft) // 2]

    # Frequency array
    frequencies = np.fft.fftfreq(len(left_channel), 1 / sample_rate)
    frequencies = frequencies[:len(frequencies) // 2]

    plt.figure(figsize=(20, 8))
    plt.title("Честотен спектър на левия и десния канал")
    plt.xlabel("Честота (Hz)")
    plt.ylabel("Амплитуда")
    plt.plot(frequencies, np.abs(left_channel_fft), label="Ляв канал", alpha=0.7)
    plt.plot(frequencies, np.abs(right_channel_fft), label="Десен канал", alpha=0.7)
    plt.legend()
    plt.xlim(0, sample_rate // 2)  # Nyquist limit
    plt.savefig(os.path.join(output_dir, "frequency_spectrum_channels.png"))
    plt.close()

def normalize_and_plot(samples, sample_rate, channels):
    """Normalize signal to 0-255 range and create visualizations"""
    # Find min and max values
    x_min = np.min(samples)
    x_max = np.max(samples)

    # Normalize signal safely
    if x_max != x_min:
        x_min_f = float(x_min)
        x_max_f = float(x_max)
        samples_f = samples.astype(np.float64)
        x_norm = np.interp(samples_f, [x_min_f, x_max_f], [0.0, 255.0]).astype(np.uint8)
    else:
        x_norm = np.zeros_like(samples, dtype=np.uint8)

    # Plot normalized signal
    duration = len(samples) / (float(sample_rate) * channels)
    time = np.linspace(0, duration, len(samples))

    plt.figure(figsize=(20, 8))
    plt.title("Нормиран аудио сигнал")
    plt.xlabel("Продължителност (s)")
    plt.ylabel("Нормирана амплитуда [0, 255]")
    plt.plot(time, x_norm)
    plt.savefig(os.path.join(output_dir, "normalized_signal.png"))
    plt.close()

    # Plot normalized histogram
    plt.figure(figsize=(20, 8))
    plt.title("Хистограма на нормализирания сигнал")
    plt.xlabel("Квантуване")
    plt.ylabel("Брой повторения")
    plt.hist(x_norm, bins=256, range=(0, 255), color='b', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "normalized_histogram.png"))
    plt.close()

def plot_full_spectrum(samples, sample_rate, channels):
    """Create complete frequency spectrum analysis"""
    # FFT transform
    fft_result = np.fft.fft(samples)

    # Frequency components and amplitudes
    frequencies = np.fft.fftfreq(len(samples), 1.0 / sample_rate)
    amplitudes = np.abs(fft_result)

    plt.figure(figsize=(20, 8))
    plt.title("Спектър на аудио сигнала")
    plt.xlabel("Честота (Hz)")
    plt.ylabel("Амплитуда")
    plt.xlim(0, sample_rate // 2)  # Nyquist limit
    plt.plot(frequencies[:len(frequencies)//2], amplitudes[:len(amplitudes)//2])
    plt.savefig(os.path.join(output_dir, "spectrum.png"))
    plt.close()

def plot_fft_result(samples, sample_rate, channels):
    """Create visualization of FFT result - magnitude spectrum"""
    # Apply FFT
    fft_result = np.fft.fft(samples)
    
    # Calculate frequency array
    frequencies = np.fft.fftfreq(len(samples), 1.0 / sample_rate)
    
    # Get only positive frequencies (up to Nyquist frequency)
    positive_freq_idx = frequencies >= 0
    positive_frequencies = frequencies[positive_freq_idx]
    fft_positive = fft_result[positive_freq_idx]
    
    # Calculate magnitude
    magnitude = np.abs(fft_positive)
    
    # Create plot
    plt.figure(figsize=(20, 8))
    plt.plot(positive_frequencies, magnitude)
    plt.title("Изображение на резултата от прилагането на бързо преобразуване на Фурие (FFT)")
    plt.xlabel("Честота (Hz)")
    plt.ylabel("Магнитуда")
    plt.xlim(0, sample_rate // 2)  # Nyquist limit
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "fft_result.png"))
    plt.close()

def plot_audio_fft_analysis(audio_file_path):
    """
    Perform FFT analysis on audio file and create time and frequency domain plots.
    
    Args:
        audio_file_path: Path to the audio file (WAV format)
    """
    # Read the audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)

    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) == 2:
        audio_data = audio_data.mean(axis=1)

    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Perform FFT
    n = len(audio_data)
    fft_result = fft(audio_data)
    frequencies = fftfreq(n, 1/sample_rate)

    # Get magnitude spectrum (only positive frequencies)
    magnitude = np.abs(fft_result)
    positive_freqs = frequencies[:n//2]
    positive_magnitude = magnitude[:n//2]

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain plot
    time = np.arange(n) / sample_rate
    ax1.plot(time, audio_data)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Signal in Time Domain')
    ax1.grid(True)

    # Frequency domain plot
    ax2.plot(positive_freqs, positive_magnitude)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT - Frequency Spectrum')
    ax2.set_xlim(0, 5000)  # Focus on 0-5kHz range
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the plot to a new output image
    output_path = os.path.join(output_dir, "audio_fft_analysis.png")
    plt.savefig(output_path)
    plt.close()

    # Print some statistics
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio duration: {n/sample_rate:.2f} seconds")
    print(f"Number of samples: {n}")
    print(f"\nDominant frequency: {positive_freqs[np.argmax(positive_magnitude)]:.2f} Hz")
    print(f"Maximum magnitude: {np.max(positive_magnitude):.2f}")
    
    return output_path

def main():
    """Main function to run the complete audio analysis"""
    print("Loading audio file...")
    audio, samples, sample_rate, channels = load_audio("./my_voice_data.mp3")

    print(f"Audio info: Sample rate={sample_rate}Hz, Channels={channels}, Duration={len(audio)/1000:.2f}s")

    # Separate channels
    left_channel = samples[::2]  # Left channel
    right_channel = samples[1::2]  # Right channel

    print("Generating visualizations...")

    # Create all visualizations
    plot_channels_comparison(left_channel, right_channel, sample_rate)
    plot_waveform(samples, sample_rate, channels)
    plot_signal_histogram(samples)
    plot_frequency_spectrum(left_channel, right_channel, sample_rate)
    normalize_and_plot(samples, sample_rate, channels)
    plot_full_spectrum(samples, sample_rate, channels)
    plot_fft_result(samples, sample_rate, channels)

    print(f"Analysis complete! All plots saved to '{output_dir}/' directory")

if __name__ == "__main__":
    main()