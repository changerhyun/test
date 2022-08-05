from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, suppress=True)
from tqdm import tqdm
import os, glob
import cv2


#########arguments##########
img_path = 'feature_output'
tsne_name = 'tsne_of_mel'
#########arguments##########

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_feature(img_path):
	img_features = []
	record_location = []
	anomaly_status = []
	filenames = glob.glob(os.path.join(img_path, '*.png'))

	for filename in tqdm(filenames):
		img = cv2.imread(filename)
		img = cv2.resize(img, (100,100))
		x = np.array(img, dtype=np.float32)/255.0

		loca = filename.split(os.path.sep)[-1].split('_')[1]
		if loca=='AV':
			loca_id=0
			loca_shorten='A'
		elif loca=='PV':
			loca_id=1
			loca_shorten='P'
		elif loca=='TV':
			loca_id=2
			loca_shorten='T'
		elif loca=='MV':
			loca_id=3
			loca_shorten='M'
		else:
			loca_id=4
			loca_shorten='Phc'

		feat = x.flatten()+loca_id
		ano = filename.split(os.path.sep)[-1].split('_')[2].replace('.png','')

		img_features.append(feat)
		record_location.append(loca_shorten)
		anomaly_status.append(ano)

	return img_features,record_location,anomaly_status


color_dict = {
	'A': 'red',
	'P': 'limegreen',
	'T': 'royalblue',
	'M': 'gold',
	'Phc': 'black'
	}

color_list = (
	'red',
	'limegreen',
	'royalblue',
	'gold',
	'black'
	)


print('[INFO] feature extracting...')
img_features,record_location,anomaly_status=get_feature(img_path)
img_features_scatter = TSNE(n_components=2, init='random', perplexity=20).fit_transform(img_features)
# img_features_scatter = PCA(n_components=2).fit_transform(img_features)
img_features_scatter = scale_minmax(img_features_scatter)


print('[INFO] visualizing...')
font_size = 5
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
for i in range(img_features_scatter.shape[0]):
	ax1.text(img_features_scatter[i,0], img_features_scatter[i,1], record_location[i], 
				color=color_dict[record_location[i]], fontdict={'weight': 'bold', 'size': font_size})
	if anomaly_status[i]=='Normal':
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(record_location[i]),
			color='blue', fontdict={'weight': 'bold', 'size': font_size}, alpha=0.4)
	else:
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(record_location[i]),
			color='red', fontdict={'weight': 'bold', 'size': font_size}, alpha=0.4)
ax1.set_title('tSNE of record location')
ax2.set_title('tSNE of normal and abnormal')
ax1.set_axis_off()
ax2.set_axis_off()
fig.savefig(tsne_name)
