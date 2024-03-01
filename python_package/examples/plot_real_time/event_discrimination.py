#!/usr/bin/env python
#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from mne.decoding import SlidingEstimator
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import make_pipeline

import sys
from scipy import signal
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf


def calculate_fbcsp(epochs_train):
    filtered = []
    for i in range(4, 45, 4):
        sos = signal.butter(2, [i, i+4], 'bandpass', fs=epochs.info['sfreq'], output='sos')
        filtered.append(signal.sosfilt(sos, epochs_train))
    filtered = np.array(filtered)
    csps = []
    n_components = 5
    csp_filters = []
    for i in range(len(filtered)):
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        features = csp.fit_transform(filtered[i], y)
        csp_filters.append(csp)
        features = features.T
        csps.append(features)
    csps = np.array(csps)
    csps = csps.reshape((len(csps)*n_components, csps.shape[-1]))
    csps = csps.T
    with open(f'{subj}_csp_filters.pickle', 'wb') as f:
        pickle.dump(csp_filters, f)

    return csps

# %%
subj = 'NSU3'
raw = mne.io.read_raw_edf(sys.argv[1], preload=True)
ch_dict = {
    'ch1': 'FCz',
    'ch2': 'Cz',
    'ch3': 'FC1',
    'ch4': 'FC2',
    'ch5': 'C1',
    'ch6': 'C2',
    'ch7': 'C3',
    'ch8': 'C4'
}

# Set montage
'''ch_dict = {
    'EEG-16': 'POz',
    'EEG-15': 'P2',
    'EEG-14': 'P1',
    'EEG-13': 'CP4',
    'EEG-12': 'CP2',
    'EEG-11': 'CPz',
    'EEG-10': 'CP1',
    'EEG-9': 'CP3',
    'EEG-8': 'C6',
    'EEG-C4': 'C4',
    'EEG-7': 'C2',
    'EEG-Cz': 'Cz',
    'EEG-6': 'C1',
    'EEG-C3': 'C3',
    'EEG-5': 'C5',
    'EEG-4': 'FC4',
    'EEG-3': 'FC2',
    'EEG-2': 'FCz',
    'EEG-1': 'FC1',
    'EEG-Fz': 'Fz',
    'EEG-0': 'FC3',
    'EEG-Pz': 'Pz',
    'EOG-left': 'EOG-l',
    'EOG-central': 'EOG-c',
    'EOG-right': 'EOG-r'
}
raw.rename_channels(ch_dict)
raw.set_channel_types({'EOG-l':'eog', 'EOG-c':'eog', 'EOG-r':'eog'})
standard_1020 = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(standard_1020)
layout_from_raw = mne.channels.make_eeg_layout(raw.info)
raw.crop(520, None)'''
raw.rename_channels(ch_dict)
standard_1020 = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(standard_1020)
evts = mne.read_events(sys.argv[2])
evt_desc = { 769:'left',  770:'right'}
annot_from_events = mne.annotations_from_events(
    events=evts, event_desc=evt_desc, sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date'])
raw.set_annotations(annotations=annot_from_events)
# Test with 6 Central Channels
#drop_chans = [channel for channel in raw.ch_names if channel not in ['C1', 'C2', 'Cz', 'CPz', 'CP1', 'CP2']]
#raw.drop_channels([channel for channel in raw.ch_names if channel not in ['C1', 'C2', 'Cz', 'FCz', 'FC1', 'FC2', 'FC4', 'FC3']])
#print(raw.ch_names)
#raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
print(raw.annotations[0])
#event_id = dict(rs=5, mi=6)#, right=7, foot=8, tongue=9)
#event_id = dict(hands=6, other=7)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
tmin, tmax = -3.0, 6.0
event_id=dict(left=769, right=770)
# %%

epochs = Epochs(
    raw,
    evts,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)


# %%
mi_data = epochs.copy().crop(tmin=1., tmax=2.).get_data()
rs_data = epochs.copy().crop(tmin=-2., tmax=-1.).get_data()
rs_data = rs_data[:len(rs_data)//2]
#X = np.concatenate((mi_data, rs_data), axis=0)
#y = np.array([0]*len(rs_data) + list(epochs.events[:,2]-5))
X = epochs.copy().crop(tmin=1., tmax=2.).get_data()
y = epochs.events[:, 2] - 768
# print(y)
X = np.concatenate((X, rs_data), axis=0)
y = np.concatenate((y, np.array([0]*len(rs_data))), axis=0)

print(X.shape)

# %%
#epochs_train = epochs.copy().crop(tmin=0., tmax=1.).get_data()
#y = epochs.events[:, -1] - 5
fmin = 2.
fmax = 20.
#fbcsp_features = calculate_fbcsp(epochs_train)



# %%

# %%

# %%
fbcsps = calculate_fbcsp(X)
X_train, X_test, y_train, y_test = train_test_split(fbcsps, y, test_size=0.2, random_state=42)
cv = ShuffleSplit(10, test_size=0.8, random_state=42)
cv_split = cv.split(X_train)

# %%

## Assemble a classifier
#rf = GridSearchCV(
#    RandomForestClassifier(),
#    param_grid={
#        'n_estimators': [10, 50, 70, 100, 150, 200],
#        'criterion': ["gini", "entropy", "log_loss"],
#        'max_features': ['sqrt', 'log2', None]
#    }
#)
#lda = GridSearchCV(
#    LinearDiscriminantAnalysis(),
#    param_grid={
#        "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#        "solver": ['eigen', 'lsqr']
#    }
#)
#svc = GridSearchCV(
#    SVC(),
#    param_grid={
#        "C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
#        "gamma": np.logspace(-2, 2, 5),
#        "kernel": ["rbf", "poly"],
#        "degree": [2, 3, 4, 5, 6]
#    }
#)
#csp = CSP(n_components=15, reg=None, log=True, norm_trace=False)
# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("LDA", LinearDiscriminantAnalysis())])


# %%
#clf = make_pipeline([ csp,  svm])
scores = cross_val_score(clf, np.array(X_train), y_train, cv=cv, n_jobs=8)

# Printing the results
class_balance = np.mean(y == y[0])
# class_balance = max(class_balance, 1.0 - class_balance)
print(
    "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
)
#a = input()
# %%
y_pred = clf.fit(X_train, y_train).score(X_test, y_test)
print(y_pred)
with open(f'MI_discrimination_{subj}.pkl','wb') as f:
    pickle.dump(clf,f)
# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#y_pred = clf.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)

#cm_display = ConfusionMatrixDisplay(cm).plot()

# %%
y_pred = clf.predict(X_test)

# %%
with open(f'MI_discrimination_{subj}.pkl','wb') as f:
    pickle.dump(clf,f)

# %%
#with open('MI_discrimination_A01T.pkl', 'rb') as f:
#    clf = pickle.load(f)


# %%



