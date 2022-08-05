import matplotlib.pyplot as plt
import librosa
import librosa.display
import os, glob
import numpy as np
from tqdm import tqdm
import skimage.io
from scipy.fft import fft
import pandas as pd


def scale_minmax(X, min=0.0, max=1.0):
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max - min) + min
	return X_scaled

def visualize(y, sr, n_fft, out_name):
	fig, ax = plt.subplots()

	### spectrum
	# N = len(y)
	# n = np.arange(N)
	# T = N/sr
	# freq = n/T
	# Y = fft(y)
	# ax.plot(freq, np.abs(Y))
	# ax.set_xlabel('Feeq(Hz)')
	# ax.set_ylabel('Amplitude')
	# ax.axis(xmin=0,xmax=sr/2)
	# ax.set_yscale('log')

	## spectrogram
	D = np.abs(librosa.stft(y=y, n_fft=n_fft))
	D = librosa.amplitude_to_db(D, ref=np.max)
	librosa.display.specshow(D,ax=ax, sr=sr, n_fft=n_fft, x_axis='time', y_axis='log')

	### mel-spectrogram
	# D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft)
	# D = librosa.power_to_db(D, ref=np.max)
	# librosa.display.specshow(D, x_axis='time', y_axis='mel', sr=sr, ax=ax)

	### mfcc
	# D = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft)
	# librosa.display.specshow(D,ax=ax, sr=sr, x_axis='time')

	### chroma
	# D = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
	# librosa.display.specshow(D,ax=ax, sr=sr, x_axis='time', y_axis='chroma')

	### spectral_contrast
	# D = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, fmin=60, n_bands=5)
	# librosa.display.specshow(D,ax=ax, sr=sr, x_axis='time')


	fig.savefig(out_name)
	plt.clf()
	plt.close()

def spectrogram(y, sr, n_fft):
	D = np.abs(librosa.stft(y=y, n_fft=n_fft))
	D = np.log(D + 1e-9)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = np.flip(img, axis=0) 
	img = 255-img
	return img

def mel(y, sr, n_fft):
	mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, win_length=n_fft)
	mels = np.log(mels + 1e-9) # add small number to avoid log(0)
	img = scale_minmax(mels, 0, 255).astype(np.uint8) # scale to fit inside 8-bit range
	img = np.flip(img, axis=0) # put low frequencies at the bottom in image
	img = 255-img # make black = more energy
	return img

def mfcc(y, sr, n_fft):
	D = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, n_mfcc=40)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

def chroma(y, sr, n_fft):
	D = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
	# D = librosa.feature.chroma_cqt(y=y, sr=sr)
	# D = librosa.feature.chroma_cens(y=y, sr=sr)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

def spectral_contrast(y, sr, n_fft):
	D = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, fmin=60, n_bands=5)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img


def melchroma(y, sr, n_fft):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2
	img = np.concatenate((D1,D2), axis=0)
	return img


#########arguments##########
feature_type = 'mel'
fft_size = 512
csv_dir = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv'
input_dir = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'
# csv_dir = 'dataset/training_data.csv'
# input_dir = 'dataset/training_data'
output_dir = 'feature_output/'

#########arguments##########

if not os.path.isdir(output_dir): os.mkdir(output_dir)
df = pd.read_csv(csv_dir)

filenames = glob.glob(os.path.join(input_dir, '*.wav'))
for filename in tqdm(filenames):
	patient_id = filename.split(os.path.sep)[-1].split('_')[0].replace('.wav', '')
	record_location = filename.split(os.path.sep)[-1].split('_')[1].replace('.wav', '')
	anomaly_status = df.loc[(df['Patient ID']==int(patient_id)), 'Outcome'].to_numpy()[0]

	out_name = output_dir+patient_id+'_'+record_location+'_'+anomaly_status+'.png'

	signalData, sr = librosa.load(filename, sr=None)
	# signalData = signalData[0:3*sr]  # 3sec

	if feature_type == 'visualize':
		visualize(signalData, sr=sr, n_fft=fft_size, out_name=out_name)
		continue
	elif feature_type == 'spectrogram':
		img = spectrogram(signalData, sr=sr, n_fft=fft_size)
	elif feature_type == 'mel':
		img = mel(signalData, sr=sr, n_fft=fft_size)
	elif feature_type == 'chroma':
		img = chroma(signalData, sr=sr, n_fft=fft_size)
	elif feature_type == 'mfcc':
		img = mfcc(signalData, sr=sr, n_fft=fft_size)
	elif feature_type == 'spectral_contrast':
		img = spectral_contrast(signalData, sr=sr, n_fft=fft_size)
	elif feature_type == 'melchroma':
		img = melchroma(signalData, sr=sr, n_fft=fft_size)

	skimage.io.imsave(out_name, img)