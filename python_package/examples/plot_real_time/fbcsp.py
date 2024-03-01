import numpy as np
from mne.decoding import CSP
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, fmin=4, fmax=45, fstep=4, sfreq=None, **kwargs) -> None:
        self.kwargs = kwargs
        self._check_sfreq(sfreq)
        self.fmin = fmin
        self.fmax = fmax
        self.fstep = fstep
        self.sfreq = sfreq
        self.csp = CSP(**kwargs)
        self.csps = []

    def fit(self, X, y=None):
        filtered = self._filter_X(X)
        #self.csps = super().fit(X,y)
        for i in range(len(filtered)):
            self.csps.append(self.csp.fit(filtered[i], y))
            #break
        return self
    
    def transform(self,X):
        filtered = self._filter_X(X)
        transformed = []
        for i in range(len(filtered)):
            temp = self.csps[i].transform(filtered[i])
            transformed.append(temp)
            #break
        transformed = np.array(transformed)
        transformed = transformed.transpose([
            1,
            0,
            2,
        ])
        print(transformed.shape)
        transformed = transformed.reshape(
            transformed.shape[0],
            transformed.shape[1] * transformed.shape[2]
        )
        return transformed

    def _check_sfreq(self, sfreq):
        """Checks if sampling frequency is provided."""
        if not isinstance(sfreq, (float, int)):
            raise ValueError("sfreq should be of type int or float (got %s)." % type(sfreq))
        if sfreq < 0 or sfreq > 2048:
            raise ValueError("sfreq should be between 0 and 2048 (got %s)" % sfreq)

    def _filter_X(self, X):
        filtered = []
        for i in range(self.fmin, self.fmax, self.fstep):
            sos = signal.butter(2, [i, i+4], 'bandpass', fs=self.sfreq, output='sos')
            filtered.append(signal.sosfilt(sos, X))
            #break
        
        return filtered