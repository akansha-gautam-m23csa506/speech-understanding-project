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

if __name__ == '__main__':
    # This is where the noise samples will be placed.
    noise_samples = 'noise_samples'
    if not os.path.exists(noise_samples):
        os.makedirs(noise_samples)

    # Download and unzip the clean audio file.
    if not os.path.exists(noise_samples + '/rirs_noises.zip'):
        rirs_noises_url = 'https://www.openslr.org/resources/28/rirs_noises.zip'
        rirs_noises_path = wget.download(rirs_noises_url, noise_samples)
        print(f"Dataset downloaded at: {rirs_noises_path}")
    else:
        print("Zipfile already exists.")
        rirs_noises_path = noise_samples + '/rirs_noises.zip'

    from zipfile import ZipFile

    if not os.path.exists(noise_samples + '/RIRS_NOISES'):
        try:
            with ZipFile(rirs_noises_path, "r") as zipObj:
                zipObj.extractall(noise_samples)
                print("Extracting noise data complete")
            # Convert 8-channel audio files to mono-channel
            wav_list = glob.glob(noise_samples + '/RIRS_NOISES/**/*.wav', recursive=True)
            for wav_path in wav_list:
                mono_wav_path = wav_path[:-4] + '_mono.wav'
                cmd = f"sox {wav_path} {mono_wav_path} remix 1"
                subprocess.call(cmd, shell=True)
            print("Finished converting the 8-channel noise data .wav files to mono-channel")
        except Exception:
            print("Not extracting. Extracted noise data might already exist.")
    else:
        print("Extracted noise data already exists. Proceed to the next step.")

    # Let's create the following list of noise samples to better showcase the effect of SNR in synthesizing noisy audio files.
    noise_sample_list = [
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0057.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0113.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0232.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0532.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0533.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0603.wav',
        'noise_samples/RIRS_NOISES/pointsource_noises/noise-free-sound-0605.wav',
    ]

    # This is where the clean audio files will be placed.
    clean_audio = 'clean_audio'
    if not os.path.exists(clean_audio):
        os.makedirs(clean_audio)

    # Download and untar the clean audio file.
    if not os.path.exists(clean_audio + '/an4_sphere.tar.gz'):
        an4_url = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'  # for the original source, please visit http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz
        an4_path = wget.download(an4_url, clean_audio)
        print(f"Dataset downloaded at: {an4_path}")
    else:
        print("Tarfile already exists.")
        an4_path = clean_audio + '/an4_sphere.tar.gz'

    # if os.path.exists(clean_audio_data + '/an4/'):
    # Untar and convert `.sph` to `.wav` (using SoX).
    tar = tarfile.open(an4_path)
    tar.extractall(path=clean_audio)

    print("Converting .sph to .wav...")
    sph_list = glob.glob(clean_audio + '/an4/**/*.sph', recursive=True)
    for sph_path in sph_list:
        wav_path = sph_path[:-4] + '.wav'
        cmd = ["sox", sph_path, wav_path]
        subprocess.run(cmd)
    print("Finished conversion.\n******")

    clean_audio_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clean_audio) for f in filenames if
                        os.path.splitext(f)[1] == '.wav']

    # Create a directory where we put the synthesized noisy files.
    noisy_files = "noisy_files"
    if not os.path.exists(noisy_files):
        os.makedirs(noisy_files)

    # Let's randomly select one clean audio and one noise sample.
    c_size = len(clean_audio_list) - 1
    n_size = len(noise_sample_list) - 1

    idx_c = random.randint(0, c_size)
    idx_n = random.randint(0, n_size)

    # Now, let's mix the selected clean audio and noise sample at 0dB SNR.
    SNR = 0
    clean_f_name = clean_audio_list[idx_c]
    noise_sample_f_name = noise_sample_list[idx_n]

    clean, fs = audioread(clean_f_name)
    noise, n_fs = audioread(noise_sample_f_name)
    if len(noise) > len(clean):
        noise = noise[0:len(clean)]
    elif len(noise) < len(clean):
        noise = concatenate_noise_sample(noise, n_fs, clean.size)

    file_name = os.path.basename(clean_f_name)
    # noisy_f_name = noisy_files + "/" + file_name[:-4] + "_0dB_snr.wav"
    #
    # clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR)
    # audiowrite(noisy_snr, fs, noisy_f_name, norm=False)
    # print("Finished creating noisy file.\n******")

    # Let's mix the files at 15dB SNR
    SNR = 15
    noisy_f_name = noisy_files + "/" + file_name[:-4] + "_15dB_snr.wav"
    clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR)
    audiowrite(noisy_snr, fs, noisy_f_name, norm=False)
    print("Finished creating noisy file.\n******")
