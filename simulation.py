import numpy as np
import bisect
from scipy.signal import fftconvolve

class ShotNoise(object):

    def __init__(self, impulse_response=None, intensity=None, marks_density=None,
                 mpp=None, signal=None, frequency=None, signal_scale=None):
        self._impulse_response = impulse_response
        self._intensity = intensity
        self._marks_density = marks_density
        self._interval_events = None
        self._all_marks = None
        self._mpp = mpp
        self._signal = signal
        self._frequency = frequency
        self._unit_in_s = None
        self._signal_scale = signal_scale
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
        self._frequency = 1 / sampling_time_s
        self._intensity = self.intensity / self.frequency
        ns = int(np.floor(duration_s * self.frequency))
        self.simulate_mpp(ns + self.truncation)
        h_eval = self.eval_kernel(sampling_time_s, ns + self.truncation)
        shot = fftconvolve(h_eval, self.innovations, mode='full')
        self._signal = shot[self.truncation : ns + self.truncation]
        self._signal_scale = np.linspace(0, duration_s, ns)
        return self.signal

    def simulate_mpp(self, length):
        if not self._interval_events and not self._all_marks:
            self._interval_events = np.random.poisson(self.intensity, size=int(length))
            self._all_marks = [self.marks_density(evt) for evt in self.interval_events]

    @property
    def innovations(self):
        return [np.sum(evt_mark) for evt_mark in self._all_marks]

    @property
    def marks(self):
        return list(self.mpp['marks'])

    @property
    def times(self):
        return list(self.mpp['times'])

    @property
    def unit_in_s(self):
        return 1 / self.frequency

    @property
    def times_s(self):
        return [index * self.unit_in_s for index in self.times]

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
    def frequency(self):
        return self._frequency

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

    def __getitem__(self, key):
        """ Returns a new ShotNoise with the part of signal specified by `key`."""
        start = int(key.start * self.frequency) if key.start else 0
        end = int(key.stop * self.frequency) if key.stop else len(self.signal) - 1
        mpp_begin = bisect.bisect_right(self.times, start)
        mpp_end = bisect.bisect_left(self.times, end)
        new_times = [time - start for time in self.times[mpp_begin : mpp_end]]
        sub_mpp = {'times' : new_times, 'marks' : self.marks[mpp_begin : mpp_end]}
        return ShotNoise(impulse_response=self.impulse_response, intensity=self.intensity, 
                         marks_density=self.marks_density, signal=self.signal[start : end], 
                         frequency=self.frequency, mpp=sub_mpp, signal_scale=self.signal_scale[start : end])



