import math
import brian2 as b2
from brian2 import * 
from brian2tools import *
from brian2 import collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurodynex3 as nd3
import spectrum
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory, plot_tools
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import random
%matplotlib inline
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import plot_tools, spike_tools
from neurodynex3.tools import plot_tools, input_factory
from brian2 import NeuronGroup, Synapses, PoissonInput, PoissonGroup, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
import tqdm
import pickle
import logging
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
logging.getLogger(b2.__name__).setLevel(logging.ERROR)
logging.getLogger(nd3.__name__).setLevel(logging.ERROR)

def get_population_activity_power_spectrum(
        rate_monitor, delta_f, k_repetitions, T_init=100*b2.ms, subtract_mean_activity=False):
    """
    Computes the power spectrum of the population activity A(t) (=rate_monitor.rate)

    Args:
        rate_monitor (RateMonitor): Brian2 rate monitor. rate_monitor.rate is the signal being
            analysed here. The temporal resolution is read from rate_monitor.clock.dt
        delta_f (Quantity): The desired frequency resolution.
        k_repetitions (int): The data rate_monitor.rate is split into k_repetitions which are FFT'd
            independently and then averaged in frequency domain.
        T_init (Quantity): Rates in the time interval [0, T_init] are removed before doing the
            Fourier transform. Use this parameter to ignore the initial transient signals of the simulation.
        subtract_mean_activity (bool): If true, the mean value of the signal is subtracted. Default is False

    Returns:
        freqs, ps, average_population_rate
    """
    data = rate_monitor.rate/b2.Hz
    delta_t = rate_monitor.clock.dt
    f_max = 1./(2. * delta_t)
    N_signal = int(2 * f_max / delta_f)
    T_signal = N_signal * delta_t
    N_init = int(T_init/delta_t)
    N_required = k_repetitions * N_signal + N_init
    N_data = len(data)
    # print(N_required)
    N_signal/2

    # print("N_data={}, N_required={}".format(N_data,N_required))
    if (N_data < N_required):
        err_msg = "Inconsistent parameters. k_repetitions require {} samples." \
                  " rate_monitor.rate contains {} samples.".format(N_required, N_data)
        raise ValueError(err_msg)
    if N_data > N_required:
        # print("drop samples")
        data = data[:N_required]
    # print("length after dropping end:{}".format(len(data)))
    data = data[N_init:]
    # print("length after dropping init:{}".format(len(data)))
    average_population_rate = np.mean(data)
    if subtract_mean_activity:
        data = data - average_population_rate
    average_population_rate *= b2.Hz
    data = data.reshape(k_repetitions, N_signal)  # reshape into one row per repetition (k)
    k_ps = np.abs(np.fft.fft(data))**2
    ps = np.mean(k_ps, 0)
    # normalize
    ps = ps * delta_t / N_signal  # TODO: verify: subtract 1 (N_signal-1)?
    freqs = np.fft.fftfreq(N_signal, delta_t)
    ps = ps[:int(N_signal/2)]
    freqs = freqs[:int(N_signal/2)]
    return freqs, ps, average_population_rate

def plot_population_activity_power_spectrum(freq, ps, max_freq, average_At=None, plot_f0=False):
    """
    Plots the power spectrum of the population activity A(t)

    Args:
        freq: frequencies (= x axis)
        ps: power spectrum of the population activity
        max_freq (Quantity): The data is plotted in the interval [-.05*max_freq, max_freq]
        plot_f0 (bool): if true, the power at frequency 0 is plotted. Default is False and the value is not plotted.

    Returns:
        the figure
    """
    first_idx_to_plot = 0 if plot_f0 else 1
    f = plt.figure()
    plt.plot(freq[first_idx_to_plot:], ps[first_idx_to_plot:], "b")
    plt.axvline(x=0., lw=1, color="k")
    plt.xlim([-.05*max_freq/b2.Hz, max_freq/b2.Hz])
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    if average_At is None:
        plt.title("Power Spectrum of population activity A(t).")
    else:
        plt.title("Power Spectrum of population activity A(t). Avg. rate = {}".format(np.round(average_At, 1)))
    return f

def get_averaged_single_neuron_power_spectrum(spike_monitor, sampling_frequency,
                                              window_t_min, window_t_max,
                                              nr_neurons_average=100, subtract_mean=False):
    """
    averaged power-spectrum of spike trains in the time window [window_t_min, window_t_max).
        The power spectrum of every single neuron's spike train is computed. Then the average
        across all single-neuron powers is computed. In order to limit the compuation time, the
        number of neurons taken to compute the average is limited to nr_neurons_average which defaults to 100

    Args:
        spike_monitor (SpikeMonitor) : Brian2 SpikeMonitor
        sampling_frequency (Quantity): sampling frequency used to discretize the spike trains.
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. Spikes
            before window_t_min are not taken into account (set a lower bound if you want to exclude an initial
            transient in the population activity)
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max.
        nr_neurons_average (int): Number of neurons over which the average is taken.
        subtract_mean (bool): If true, the mean value of the signal is subtracted before FFT. Default is False

    Returns:
        freq, mean_ps, all_ps_dict, mean_firing_rate, mean_firing_freqs_per_neuron
    """

    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"

    spiketrains = spike_monitor.spike_trains()
    nr_neurons = len(spiketrains)

    sample_neurons = []
    nr_samples = 0
    if nr_neurons <= nr_neurons_average:
        sample_neurons = range(nr_neurons)
        nr_samples = nr_neurons
    else:
        idxs = np.arange(nr_neurons)
        np.random.shuffle(idxs)
        sample_neurons = idxs[:(nr_neurons_average)]
        nr_samples = nr_neurons_average

    sptrs = spike_tools.filter_spike_trains(spike_monitor.spike_trains(), window_t_min, window_t_max, sample_neurons)
    time_window_size = window_t_max - window_t_min
    discretization_dt = 1./sampling_frequency
    if window_t_max is None:
        window_t_max = max(spike_monitor.t)
    vector_length = 1+int(math.ceil(time_window_size/discretization_dt))  # +1: space for rounding issues
    freq = 0
    spike_count = 0
    print(nr_samples, vector_length/2)
    all_ps = np.zeros([nr_samples, int(vector_length/2)], float)
    all_ps_dict = dict()
    mean_firing_freqs_per_neuron = dict()
    for i in range(nr_samples):
        idx = sample_neurons[i]
        vec = spike_tools._spike_train_2_binary_vector(
            sptrs[idx]-window_t_min, vector_length, discretization_dt=discretization_dt)
        ps, freq = _get_spike_train_power_spectrum(vec, discretization_dt, subtract_mean)
        all_ps[i, :] = ps
        all_ps_dict[idx] = ps
        nr_spikes = len(sptrs[idx])
        nu_avg = nr_spikes / time_window_size
        # print(nu_avg)
        mean_firing_freqs_per_neuron[idx] = nu_avg
        spike_count += nr_spikes  # count in the subsample which is filtered to [window_t_min, window_t_max]

    mean_ps = np.mean(all_ps, 0)
    mean_firing_rate = spike_count / nr_samples / time_window_size
    print("mean_firing_rate:{}".format(mean_firing_rate))
    return freq, mean_ps, all_ps_dict, mean_firing_rate, mean_firing_freqs_per_neuron

def _get_spike_train_power_spectrum(spike_train, delta_t, subtract_mean=False):
    st = spike_train/b2.ms
    if subtract_mean:
        data = st-np.mean(st)
    else:
        data = st
    N_signal = data.size
    ps = np.abs(np.fft.fft(data))**2
    # normalize
    ps = ps * delta_t / N_signal  # TODO: verify: subtract 1 (N_signal-1)?
    freqs = np.fft.fftfreq(N_signal, delta_t)
    ps = ps[:int(N_signal/2)]
    freqs = freqs[:int(N_signal/2)]
    return ps, freqs

def plot_spike_train_power_spectrum(freq, mean_ps, all_ps, max_freq,
                                    nr_highlighted_neurons=2, mean_firing_freqs_per_neuron=None, plot_f0=False):
    """
    Visualizes the power spectrum of the spike trains.

    Args:
        freq: frequencies (= x axis)
        mean_ps: average power taken over all neurons (typically all of a subsample).
        all_ps (dict): power spectra for each single neuron
        max_freq (Quantity): The x-lim of the plot is [-0.05*max_freq, max_freq]
        mean_firing_freqs_per_neuron (float): None or the mean firing rate averaged across the neurons. Default is
            None in which case the value is not shown in the legend
        plot_f0 (bool): if true, the power at frequency 0 is plotted. Default is False and the value is not plotted.
    Returns:
        the figure and the index of the random neuron for which the PS is computed: all_ps[random_neuron_index]
    """
    nr_neurons = len(all_ps)
    f = plt.figure()
    color = "r"
    msize = 10
    legend_text = []
    random_neuron_index = []

    first_idx_to_plot = 0 if plot_f0 else 1
    for i in range(nr_highlighted_neurons):
        rand_idx = np.random.randint(nr_neurons)
        print(rand_idx)
        rand_key = list(all_ps.keys())[rand_idx]
        rand_neuron_ps = all_ps[rand_key]
        plt.plot(freq[first_idx_to_plot:], rand_neuron_ps[first_idx_to_plot:], linestyle=" ", markersize=msize, c=color)
        color = [.75, .75, .75]  # print the first neuron in red and all others in gray
        msize = 8
        random_neuron_index.append(rand_key)
        if mean_firing_freqs_per_neuron is None:
            legend_text.append("PS Neuron #{}".format(rand_key))
        else:
            legend_text.append("PS Neuron #{}, avg rate={}"
                               .format(rand_key, np.round(mean_firing_freqs_per_neuron[rand_key], 1)))

    plt.plot(freq[first_idx_to_plot:], mean_ps[first_idx_to_plot:], "b")

    if mean_firing_freqs_per_neuron is None:
        legend_text.append("averaged PS")
    else:
        avg_rate = np.mean(list(mean_firing_freqs_per_neuron.values()))
        legend_text.append("averaged PS, avg rate={}".format(np.round(avg_rate, 1)))

    plt.legend(legend_text)
    plt.xlim([-0.05*max_freq/b2.Hz, max_freq/b2.Hz])
    plt.axvline(x=0., lw=1, color="k")
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.title("Single neuron power spectrum and average")
    return f, random_neuron_index

def next_power_of_two(x: int):
    """
    next power of two implemented using bit length of integer
    (equivalent to ceil(log2(x)), ie smallest sp such that x <= 2 ** sp)

    :param x: value for which the next largest power of 2 is tb determined
    :return: smallest power of two greater equal to x (smallest sp such that  x <= 2 ** sp)
    """
    # shifting x to the left by 1 for bits results in bit(e) = x = sp(e) (incl shift for bit(.)),
    #  ie bit'(x-1) = sp'(x) where bit', sp' are the respective inverse fct
    # bit -> x :             1 -> (0,1), 2 -> (2,3), 3 -> (4,5,6,7), 4 -> (8, ..., 15)
    # sp  -> x : 0 -> (0,1), 1 -> (2),   2 -> (3,4), 3 -> (5,6,7,8)
    return int(x - 1).bit_length()


def mt_psd(rate: np.ndarray, dt: float, nfft: int = None):
    """
    Power spectral density of the population rate computed with a multi taper method

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param nfft: length of the output of fft (n-point discrete, where n = nft)
                  - set only if you desire a specific nfft - defaults to 2 ** sp, where sp is smallest num for which rate.size <= 2 ** sp
    :return: frequencies and the power spectral density (at the respective frequencies)
    """
    # equivalent to internal impl (except for capping at 256) as int(np.ceil(np.log2(n))) == int(n-1).bit_length()
    # 2 ** n, where n is smallest power of 2 larger than x.size - fft works best when length is a power of 2
    if nfft == None:
        nfft = 2 ** next_power_of_two(np.size(rate))
    # multi taper
    frequency, Sk_complex, weights, _ = mt_spectrum(rate, dt, nfft)

    # compute the energy spectral density (from complex spectrum)
    Sk = abs(Sk_complex) ** 2

    # average over slepian windows using weights
    spectral = np.mean(Sk * weights, axis=0)[: nfft // 2]

    # compute power spectral density
    spectral = spectral / np.size(rate)

    return frequency, spectral
    
def mt_spectrum(rate: np.ndarray, dt: float, nfft: int = None):
    """
    spectrum of the population rate computed with a multi taper method

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param nfft: length of the output of fft (n-point discrete, where n = nft)
                  - set only if you desire a specific nfft - defaults to 2 ** sp, where sp is smallest num for which rate.size <= 2 ** sp
    :return: frequencies, complex spectrum, weights, eigenvalues of multitaper method
    """
    # equivalent to internal impl (except for capping at 256) as int(np.ceil(np.log2(n))) == int(n-1).bit_length()
    # 2 ** n, where n is smallest power of 2 larger than x.size - fft works best when length is a power of 2
    if nfft == None:
        nfft = 2 ** next_power_of_two(np.size(rate))

    # sampling frequency [Hz]
    f_s = 1000.0 / dt

    # nyquist frequency [Hz] - max frequency for which signal can be reliable reconstructed ~ f_n is unique minimum (aliasing)
    f_n = f_s / 2.0

    # fft transforms equally spaced samples to same-length equally spaced samples of freq domain at interval 1 / duration of input sequence = 1 / (n / f_s) = f_s / n
    #  - starting at 0.0 for nfft samples we have  nfft * f_s / n ~ f_s and therefore nfft/2 data points in interval [0,f_n]
    frequency = np.linspace(0.0, f_n, nfft // 2)

    # multi taper
    Sk_complex, weights, eigenvalues = spectrum.pmtm(
        rate, NW=2.5, NFFT=nfft, method="eigen"
    )
    return frequency, Sk_complex, weights, eigenvalues

def multitaper_power_spectral_density(
    rate: np.ndarray,
    dt: float,
    w_sliding: int = None,
    w_step: float = 0.1,
    nfft: int = None,
):
    """
    Power spectral density of the population rate computed with a multi taper method
    computed over the entire time series or for separate (yet overlapping) time intervals using a sliding window
    without padding when parameter w_sliding is set

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param w_sliding: sliding window used for computing psd discretized over time (without padding)
                     - when not set, defaults to computing psd over entire time series
    :param w_step: step size of the sliding window as a fraction of the sliding window size (param w_sliding) - irrelevant when w_sliding not set
    :param nfft: length of the output of fft (n-point discrete, where n = nfft)
                  - set only if you desire a specific nfft, eg to increase the resolution
    :return: frequencies and the power spectral density (for entire time series psd shape: (nfft/2,1); for separate intervals psd shape: (nfft/2, intervals) (at the respective frequencies)
    """

    if w_sliding == None:
        return mt_psd(rate, dt, nfft=nfft)
    else:

        if w_sliding < 1 or w_sliding > np.size(rate):
            raise ValueError(f"w_sliding must be in [1,np.size(rate)]. Is {w_sliding}")
        if w_step < 0 or w_step > 1:
            raise ValueError(
                f"w_step must be in [0,1] specifying step size as a fraction of sliding window size. Is {w_step}"
            )

        w_step = int(w_sliding * w_step)

        # +1 for intial sliding window size and then rest/w_step intervals on the rest of the sequence
        num_intervals = int((np.size(rate) - w_sliding) / w_step) + 1

        if nfft == None:
            # size of the rate signal in each time interval
            nfft = 2 ** next_power_of_two(w_sliding)

        psd = np.zeros((nfft // 2, num_intervals))
        for i in range(num_intervals):
            frequency, psd[:, i] = mt_psd(
                rate[w_step * i : w_step * i + w_sliding], dt, nfft=nfft
            )
        return frequency, psd

def smooth_rate(self, window="gaussian", width=None):
    """
    smooth_rate(self, window='gaussian', width=None)

    Return a smooth version of the population rate.

    Parameters
    ----------
    window : str, ndarray
        The window to use for smoothing. Can be a string to chose a
        predefined window(``'flat'`` for a rectangular, and ``'gaussian'``
        for a Gaussian-shaped window). In this case the width of the window
        is determined by the ``width`` argument. Note that for the Gaussian
        window, the ``width`` parameter specifies the standard deviation of
        the Gaussian, the width of the actual window is ``4*width + dt``
        (rounded to the nearest dt). For the flat window, the width is
        rounded to the nearest odd multiple of dt to avoid shifting the rate
        in time.
        Alternatively, an arbitrary window can be given as a numpy array
        (with an odd number of elements). In this case, the width in units
        of time depends on the ``dt`` of the simulation, and no ``width``
        argument can be specified. The given window will be automatically
        normalized to a sum of 1.
    width : `Quantity`, optional
        The width of the ``window`` in seconds (for a predefined window).

    Returns
    -------
    rate : `Quantity`
        The population rate in Hz, smoothed with the given window. Note that
        the rates are smoothed and not re-binned, i.e. the length of the
        returned array is the same as the length of the ``rate`` attribute
        and can be plotted against the `PopulationRateMonitor` 's ``t``
        attribute.
    """
    if width is None and isinstance(window, str):
        raise TypeError("Need a width when using a predefined window.")
    if width is not None and not isinstance(window, str):
        raise TypeError("Can only specify a width for a predefined window")

    if isinstance(window, str):
        if window == "gaussian":
            width_dt = int(np.round(2 * width / (0.1 *msecond)))
            # Rounding only for the size of the window, not for the standard
            # deviation of the Gaussian
            window = np.exp(
                -np.arange(-width_dt, width_dt + 1) ** 2
                * 1.0
                / (2 * (width / (0.1 *msecond)) ** 2)
            )
        # elif window == "flat":
        #     width_dt = int(width / 2 / 0.1 *msecond) * 2 + 1
        #     used_width = width_dt * 0.1 *msecond
        #     if abs(used_width - width) > 1e-6 * self.clock.dt:
        #         logger.info(
        #             f"width adjusted from {width} to {used_width}",
        #             "adjusted_width",
        #             once=True,
        #         )
        #     window = np.ones(width_dt)
        else:
            raise NotImplementedError(f'Unknown pre-defined window "{window}"')
    else:
        try:
            window = np.asarray(window)
        except TypeError:
            raise TypeError(f"Cannot use a window of type {type(window)}")
        if window.ndim != 1:
            raise TypeError("The provided window has to be one-dimensional.")
        if len(window) % 2 != 1:
            raise TypeError("The window has to have an odd number of values.")
    return Quantity(
        np.convolve(self.rate, window * 1.0 / sum(window), mode="same"),
        dim=hertz.dim,
    )

