import numpy as np
from scipy.signal import fftconvolve

class ShotNoise(object):

    def __init__(self, impulse_response=None, intensity=None, marks_density=None):
        self._impulse_response = impulse_response
        self._intensity = intensity
        self._marks_density = marks_density
        self._interval_events = None
        self._marks = None
        self._all_marks = None
        self._mpp = None
        self._signal = None
        self._signal_scale = None
        self._truncation = 2000

    @property
    def impulse_response(self):
        return self._impulse_response

    @impulse_response.setter
    def impulse_response(self, function):
        self._impulse_response = function

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value

    @property
    def marks_density(self):
        return self._marks_density

    @marks_density.setter
    def marks_density(self, function):
        self._marks_density = function

    def simulate(self, duration_s, sampling_time_s):
        self._interval_events = None
        self._all_marks = None
        self._mpp = None
        intensity = self._intensity * sampling_time_s
        ns = int(np.floor(duration_s / float(sampling_time_s)))
        self.simulate_mpp(intensity, ns + self.truncation)
        h_eval = self.eval_kernel(sampling_time_s, ns + self.truncation)
        shot = fftconvolve(h_eval, self.innovations, mode='full')
        self._signal = shot[self.truncation : ns + self.truncation]
        self._signal_scale = np.linspace(0, duration_s, ns)
        return self.signal

    def simulate_mpp(self, intensity, length):
        if not self._interval_events and not self._all_marks:
            self._interval_events = np.random.poisson(lam=intensity, size=int(length))
            self._all_marks = [self.marks_density(evt) for evt in self.interval_events]

    @property
    def innovations(self):
        return [np.sum(evt_mark) for evt_mark in self._all_marks]

    @property
    def marks(self):
        return self.mpp['marks']

    @property
    def times(self):
        return self.mpp['times']

    @property
    def interval_events(self):
        return self._interval_events

    @property
    def signal(self):
        return self._signal

    @property
    def truncation(self):
        return self._truncation

    @truncation.setter
    def truncation(self, value):
        self._truncation = value

    @property
    def get_mpp(self):
        mpp_list = zip(*[(index - self.truncation, mark) for index, marks in enumerate(self._all_marks) 
                          for mark in marks if index > self.truncation])
        self._mpp = {'times' : mpp_list[0], 'marks' : mpp_list[1]}

    @property
    def mpp(self):
        if self._mpp is None:
           self.get_mpp 
        return self._mpp

    @property
    def signal_scale(self):
        return self._signal_scale

    def eval_kernel(self, sampling_time_s, nb_points):
        return np.array([self.impulse_response(i * sampling_time_s) for i in range(nb_points)])





