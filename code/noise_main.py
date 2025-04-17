import os
import subprocess
import wget
import glob
import tarfile
import librosa
import soundfile as sf
import random
import numpy as np

# function to read audio
def audioread(path, norm = True, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            if rms == 0:
               rms = 1
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            if rms == 0:
               rms = 1
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr

# funtion to write audio
def audiowrite(data, fs, destpath, norm=False):
    if norm:
        eps = 0
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms+eps)
        data = data * scalar
        if max(abs(data))>=1:
            data = data/max(abs(data), eps)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, data, fs)
    return


# function to mix a clean speech with a noise sample at a specified SNR level
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean ** 2).mean() ** 0.5
    if rmsclean == 0:
        rmsclean = 1

    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean ** 2).mean() ** 0.5

    rmsnoise = (noise ** 2).mean() ** 0.5
    if rmsnoise == 0:
        rmsnoise = 1

    scalarnoise = 10 ** (-25 / 20) / rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise ** 2).mean() ** 0.5
    if rmsnoise == 0:
        rmsnoise = 1

    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10 ** (snr / 20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech

# Add zeros to a noise sample to make it of the same duration as the clean audio.
def concatenate_noise_sample(noise, fs, len_clean):
    silence_length = 0.5
    while len(noise) <= len_clean:
       noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
       noise = np.append(noiseconcat, noise)

    if noise.size > len_clean:
       noise = noise[0:len_clean]

    return noise

# Directory to save synthesized noisy files.
noisy_files = "noisy_files"
if not os.path.exists(noisy_files):
    os.makedirs(noisy_files)

# SNR value (can make this a loop for multiple SNRs if needed)
SNR = 5

# Gather all mono noise samples
noise_sample_list = glob.glob('noise_samples/RIRS_NOISES/pointsource_noises/*_mono.wav', recursive=True)
# clean_audio_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk("/Users/anchitmulye/Downloads/train") for f in filenames if
#                         os.path.splitext(f)[1] == '.wav']
clean_audio_list = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk("/Users/anchitmulye/Downloads/train")
    for f in filenames
    if f.endswith('.wav') and not f.endswith('_noise.wav')
]

for clean_path in clean_audio_list:
    clean, fs = audioread(clean_path)
    file_name_clean = os.path.splitext(os.path.basename(clean_path))[0]
    clean_dir = os.path.dirname(clean_path)

    # Pick one random noise file
    noise_path = random.choice(noise_sample_list)
    noise, n_fs = audioread(noise_path)

    # # Optional: resample noise if needed
    # if fs != n_fs:
    #     noise = librosa.resample(noise, orig_sr=n_fs, target_sr=fs)
    #     n_fs = fs

    # Match length
    if len(noise) > len(clean):
        noise = noise[:len(clean)]
    else:
        noise = concatenate_noise_sample(noise, fs, len(clean))

    # Mix with given SNR
    _, _, noisy = snr_mixer(clean, noise, SNR)

    # Save at same location
    out_file_name = f"{file_name_clean}_{SNR}_noise.wav"
    out_path = os.path.join(clean_dir, out_file_name)
    audiowrite(noisy, fs, out_path, norm=False)

    print(f"Saved: {out_path}")
