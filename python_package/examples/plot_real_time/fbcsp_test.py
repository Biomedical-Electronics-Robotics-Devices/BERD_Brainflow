import mne
import json
import pickle
import numpy as np
import mne
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import sys
from scipy import signal
from mne import Epochs, pick_types
from mne.decoding import CSP

from mne.decoding import CSP
from fbcsp import FBCSP
from sklearn.svm import SVC


def read_channel_dictionary(fname):

    with open(fname, 'r') as f:
        channel_dictionary = json.loads(f.read())

    return channel_dictionary


raw = mne.io.read_raw_edf('23_02_2024__13_39_56.edf', preload=True)
evts = mne.read_events('23_02_2024__13_39_56.edf_events.evt')
evt_desc = { 769:'left',  770:'right'}
annot_from_events = mne.annotations_from_events(
events=evts, event_desc=evt_desc, sfreq=raw.info['sfreq'],
orig_time=raw.info['meas_date'])
raw.set_annotations(annot_from_events)
ch_dict = read_channel_dictionary('channel_dictionary.json')
raw.rename_channels(ch_dict)
standard_1020 = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(standard_1020)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
tmin, tmax = -3.0, 6.0
event_id=dict(left=769, right=770)
# %%

epochs = mne.Epochs(
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


# Get data
mi_data = epochs.copy().crop(tmin=1., tmax=2.).get_data()
rs_data = epochs.copy().crop(tmin=-2., tmax=-1.).get_data()
rs_data = rs_data[:len(rs_data)//2]     # Balance resting state with the others

# Create data and label arrays
X = np.concatenate((mi_data, rs_data), axis=0)
#y = np.array([0]*len(rs_data) + list(epochs.events[:,2]-5))
#X = epochs.copy().crop(tmin=1., tmax=2.).get_data()
y = epochs.events[:, 2] - 768
y = np.concatenate((y, np.array([0]*len(rs_data))), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = ShuffleSplit(10, test_size=0.8, random_state=42)
cv_split = cv.split(X_train)
fbcsp = FBCSP(sfreq=raw.info['sfreq'], n_components=5, reg=None, log=True, norm_trace=False)
#fbcsp = CSP( n_components=5, reg=None, log=True, norm_trace=False)
fbcsp.fit(X_train, y_train)
l = fbcsp.transform(X_train)
print(l.shape)
lda = SVC()
#csp.fit(X=X_train, y=y_train)
clf = Pipeline([("FBCSP", fbcsp), ("LDA", lda)])
scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=None)
print(scores)
print(l.shape)
class_balance = np.mean(y == y[0])
print(
    "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
)