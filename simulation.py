import numpy as np
import pp
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

def kernel_evaluation(h , ts, size):
    """Function which evaluates the function h at points i*ts for each component i of the vector size   
    """
    m = size[0]
    return np.array([h(i * ts) for i in size])


def simul_mpp_homogeneous (intensity, marks, length):
    """Function that simulates the increments of the background Levy process for a given ts. In the shot-noise setting,
    this function is carried out in two steps:
    - simulation of 'length' random numbers following a Poisson process with rate intensity*ts. We denote 
    - sum of N simulations of i.i.d. r.v.'s following the law specified by marks
    """
    poisson_random_numbers = np.random.poisson(lam=intensity, size=int(length) + 2)
    N = np.sum(poisson_random_numbers) + 2
    U = np.random.rand(N) <= .7
    Y = U * (.3 + .05 * np.random.randn(N)) + (1 - U) * (.7 + .03 * np.random.randn(N))
    Y = np.cumsum(Y)
    poisson_random_numbers = np.cumsum(poisson_random_numbers)   
    Z = Y[poisson_random_numbers]
    del Y 
    levy_innov = np.diff(Z)

    return levy_innov[1:], poisson_random_numbers


def simul_mpp_compton (intensity, marks, length):
    """ Same as simul_mpp_homogeneous with a different mark's density in order to simulate a mark's density that
    model Compton scattering.
    Function that simulates the increments of the background Levy process for a given ts. In the shot-noise setting,
    this function is carried out in two steps:
    - simulation of 'length' random numbers following a Poisson process with rate intensity*ts. We denote 
    - sum of N simulations of i.i.d. r.v.'s following the law specified by marks
    """
    
    poisson_random_numbers = np.random.poisson(lam=intensity, size=int(length) + 2)
    N = np.sum(poisson_random_numbers) + 2
    U=np.random.rand(N)
    Y = (U <= .2) * np.abs(.3 * np.random.rand(N))
    Y += (U > .2) * (U <= .6) * (0.8 + 0.02 * np.random.rand(N))
    Y += (U > .6) * (U <= .8) * (0.5 + 0.02 * np.random.rand(N))
    Y += (U > .8) * (0.65 + 0.02 * np.random.rand(N))
    Y = np.cumsum(Y)
    poisson_random_numbers = np.cumsum(poisson_random_numbers)   
    Z = Y[poisson_random_numbers]
    del Y 
    levy_innov = np.diff(Z)

    return levy_innov[1:], poisson_random_numbers


def simulation_shot_noise(intensity, marks, ts, h, time, M=10000):
    """This script simulates a sample of size ns of shot-noise process at points i*ts for 1<=i<=ns which has the following
    characteristics:
    - h : impulse response of the electronical system
    - intensity : arrival rate of events
    - marks : density of the i.i.d. marks associated to new epochs
    - ns : number of samples
    - ts : sampling time
    - M : upper bound of the number of epochs
    """
    s_intensity = intensity * ts
    ns= np.floor(time / float(ts))
    # a) truncation of the sum    
    size = np.linspace(- M, ns - 1, ns + M)
    length = size.shape[0]
    # b) Simulation of the innovations of the background Levy process   
    levy_innov, _ = simul_mpp_homogeneous (s_intensity, marks, ns + M)
    # c) Evaluation of the kernel at sampling points
    h_eval = kernel_evaluation(h, ts, size)
    # d) Discrete convolution
    y = np.convolve(h_eval, levy_innov, mode='same')
    # e) Path selection
    sn = y[M:]

    return sn


def simulation_shot_noise_compton(intensity, marks, ts, h, time, M=10000):
    """This script simulates a sample of size ns of shot-noise process at points i*ts for 1<=i<=ns which has the following
    characteristics:
    - h : impulse response of the electronical system
    - intensity : arrival rate of events
    - marks : density of the i.i.d. marks associated to new epochs
    - ns : number of samples
    - ts : sampling time
    - M : upper bound of the number of epochs
    """
    s_intensity = intensity * ts
    ns = np.floor(time / float(ts))
    # a) truncation of the sum    
    size = np.linspace(- M, ns - 1, ns + M)
    length = size.shape[0]
    # b) Simulation of the innovations of the background Levy process   
    levy_innov, _ = simul_mpp_compton (s_intensity, marks, ns + M)
    # c) Evaluation of the kernel at sampling points
    h_eval = kernel_evaluation(h, ts, size)
    # d) Discrete convolution
    y = np.convolve(h_eval, levy_innov, mode='same')
    # e) Path selection
    sn = y[M:]

    return sn


def parallel_simul_shot(intensity, marks, ts, h, n_sample, M=15000):
    """ This function extends the function 'simulation_shot_noise' by running multiple threads of the latter. 
    However, it is important to notice that the resulting points are not low frequency samples of a stationary shot-noise 
    but are low frequency samples of several stationary shot-noise that share the same finite-dimensional laws. 
    ====
    Args
    ====
    - h : impulse response of the electronical system
    - intensity : arrival rate of events
    - marks : density of the i.i.d. marks associated to new epochs
    - n_sample : number of samples ( length of the output vector)
    - ts : sampling time
    - M : upper bound of the number of epochs
    
    """
    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list 
    shot_simulated = list()
    job_server = pp.Server()
    thread_number = job_server.get_ncpus()
    nLoops = int(np.floor(n_sample / (10000 * thread_number)) + 1)
    # Send the jobs
    
    for i in range(nLoops):
        jobs = []
        for arg in range(thread_number):
            jobs.append(job_server.submit(func=simulation_shot_noise,
                                          args=(intensity,
                                                marks, ts,
                                                h, int(ts*1e4),
                                                M,),
                                                modules=('numpy as np',
                                                         'from simulation import simul_mpp_homogeneous, kernel_evaluation')))
        for job in jobs:
            shot_simulated.append(job())
    job_server.destroy()
    sn = np.array(shot_simulated) 
    sn = sn.reshape((sn.shape[0] * sn.shape[1], 1))

    return sn[0:n_sample - 1]


def parallel_simul_shot_compton(intensity, marks, ts, h, n_sample, M=15000):
    """ This function extends the function 'simulation_shot_noise' by running multiple threads of the latter. 
    However, it is important to notice that the resulting points are not low frequency samples of a stationary shot-noise 
    but are low frequency samples of several stationary shot-noise that share the same finite-dimensional laws. 
    ====
    Args
    ====
    - h : impulse response of the electronical system
    - intensity : arrival rate of events
    - marks : density of the i.i.d. marks associated to new epochs
    - n_sample : number of samples ( length of the output vector)
    - ts : sampling time
    - M : upper bound of the number of epochs
    
    """
    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list 
    shot_simulated = list()
    job_server = pp.Server()
    thread_number = job_server.get_ncpus()
    nLoops = int(np.floor(n_sample / (10000 * thread_number)) + 1)
    # Send the jobs
    
    for i in range(nLoops):
        jobs = []
        for arg in range(thread_number):
            jobs.append(job_server.submit(func=simulation_shot_noise_compton, 
                                          args=(intensity, 
                                                marks, ts,
                                                h, int(ts*1e4),
                                                M,),
                                                modules=('numpy as np',
                                                         'from simulation import simul_mpp_compton, kernel_evaluation')))
        for job in jobs:
            shot_simulated.append(job())
    job_server.destroy()
    sn = np.array(shot_simulated) 
    sn = sn.reshape((sn.shape[0] * sn.shape[1], 1))

    return sn[0:int(n_sample - 1)]
    

def simulation_shot_noise_with_underlying_PP(intensity, marks, ts, h, time, M=10000):
    """This script simulates a sample of size ns of shot-noise process at points i*ts for 1<=i<=ns which has the following
    characteristics:
    ====
    Args
    ====
    - h : impulse response of the electronical system
    - intensity : arrival rate of events
    - marks : density of the i.i.d. marks associated to new epochs
    - ns : number of samples
    - ts : sampling time
    - M : upper bound of the number of epochs
    =======
    Returns
    =======
    - sn : low frequency sample path of the shot-noise
    - output : the levy innovations. It is exactly equal to the underlying marked point process when the
    sampling time ts is sufficently low.
    """
    s_intensity = intensity * ts
    ns= np.floor(time / float(ts))
    # a) truncation of the sum    
    size = np.linspace(- M, ns - 1, ns + M)
    length = size.shape[0]
    # b) Simulation of the innovations of the background Levy process   
    levy_innov, poisson_random_numbers = simul_mpp_homogeneous(s_intensity, marks, ns + M)
    # c) Evaluation of the kernel at sampling points
    h_eval = kernel_evaluation(h, ts, size)
    # d) Discrete convolution
    y = np.convolve(h_eval, levy_innov, mode='same')
    # e) Path selection
    sn = y 
    threshold = np.abs(ns - M) / 2
    levy_innov = levy_innov[threshold:]
    output = (levy_innov != 0).astype(int)
    output = np.concatenate((output, np.zeros(threshold)), axis=0)
    
    return sn, output
        