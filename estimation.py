import patsy as pt
import numpy as np
import pp
import scipy.optimize as opt
import time
import math
import design_operation as fdp
from cvxopt import solvers, matrix
from scipy.signal import argrelextrema


def _compute_b_spline_basis(imp_res, nb_knots, supp=[0,1], pas=0.0005, supp_h=4, bins_h=4000):
    """
    This function first computes the design matrix given in section 3 of the associated article. In order to optimize the computation, we use
    a parallel processing package (called pp module and available online on parallelpython.com). This is the most time-consuming step of the
    estimation procedure. If the script is intended to be used several times for the same acquisition electronical system, we propose to save the
    computation of the design matrix in a file that will precise the impulse response and the evaluation grid that was used, so as the number
    of B-splines (and jointly the support of the B-splines).
    The function can be decomposed into two steps:
    1) Computation of the B-splines given a set of knots, a degree (deg) a support (supp) and a bandwidth (bd):
    """
    # 1) B-splines:
    knots = np.linspace(supp[0],supp[1], nb_knots)
    N = min(np.floor((supp[1] - supp[0]) / pas), 1000000)
    grid = np.linspace(supp[0], supp[1], N + 1)
    pas_x = grid[1] - grid[0]
    b_spline_basis  = pt.splines.bs(grid, 3 + nb_knots)
    # 2) Scalar Product:
    b_spline_basis_m = pt.splines.bs(grid,1 + nb_knots)
    scalar_P = pas * np.transpose(b_spline_basis_m).dot(b_spline_basis_m)
    # 3) Design matrix:
    grid_t = np.linspace(0, supp_h, bins_h)
    pas_t = grid_t[1]-grid_t[0]
    h_t = imp_res(grid_t)
    pas_f = pas_x * pas_t
    return scalar_P, b_spline_basis, grid, h_t, pas_f


def _design_matrix_fortran(grid_x, grid_t, eval_grid, pas_f):
        
    def list_append(grid_x, grid_t, eval_grid, pas_f):
        return fdp.matrix_op(grid_x, grid_t, eval_grid, pas_f)

    # Send the jobs
    job_server = pp.Server()
    thread_number = job_server.get_ncpus()
    nLoops = eval_grid.shape[0] / thread_number 
    print nLoops
    sub_eval = np.zeros((thread_number, nLoops))
    for i in range(thread_number):
        a = np.linspace(0, nLoops - 1, nLoops) + i * nLoops
        sub_eval[i,:] = eval_grid[a.astype(int)]
    jobs = []
    for k in range(thread_number):
        jobs.append(job_server.submit(func=list_append,
                                      args=(grid_x, grid_t, sub_eval[k,:], pas_f,),
                                      modules=('numpy as np','design_operation as fdp',)))      
    for i,job in enumerate(jobs):
        if i == 0:
            dM = job()
        else:
            dM = np.concatenate((dM, job()))

    job_server.destroy()
    return dM


def _estimation(kappa, design_matrix, cov_estim, penalties, A, C, grid_x, b_spline_functions, step=0.0005):
    """Function that implements the estimation procedure for multiple penalty terms
    and returns the estimator chosen by the generalized cross validation.
    =====
    Args:
    =====
    - kappa : estimators of the second characteristic function
    - design_matrix : design matrix depending on the B-spline basis and the evaluation grid
    - cov_matrix : covariance matrix of kappa
    - penalties : list of penalty terms
    ========
    Outputs:
    ========
    Returns the estimators of the flow function f_lambda for each penalty term.
    """
    Kappa = np.mean(kappa, axis=0)
    D = np.transpose(A).dot(C).dot(A) * float(step ** 2)

    epsilon =1e-7
    epsilon2 = 1e-13
    print "Cholesky regularization factor : {}".format(epsilon)
    print "Regularity penalization factor :{}".format(epsilon2)

    cov_estim_2 = cov_estim + epsilon*np.eye(cov_estim.shape[0])
    W = matrix(np.linalg.inv(cov_estim_2)); W =.5*(W+W.T)

    S = np.sum(b_spline_functions,axis=0)
    A1 = matrix(np.concatenate([np.real(design_matrix), np.imag(design_matrix)]))
    b1 = np.concatenate([np.real(Kappa), np.imag(Kappa)]).reshape(A1.size[0], )

    err1 = np.zeros(len(penalties)) ; err2 = np.zeros(len(penalties)) ; lambda_estim = np.zeros(len(penalties))
    err = np.zeros(len(penalties)) ; err3 = np.zeros(len(penalties))
    theta = np.zeros((len(penalties), b_spline_functions.shape[1]))
    spectrum =np.zeros((len(penalties), grid_x.shape[0]))

    q = matrix(-(A1.T * W.T) * matrix(b1))
    G = - matrix(b_spline_functions[::1,:]) 
    h = matrix(np.zeros(G.size[0]))
    for i,pen in enumerate(penalties):
        P = A1.T * W.T * A1 + pen * np.matrix(D)   
        P = matrix(P)
        shot_solv = solvers.qp(.5 * (P + P.T), q, G, h)
        theta[i,:] = np.asarray(shot_solv['x'] ).reshape(-1)
        err1[i] = np.sum(np.abs((np.array(A1 * matrix(theta[i,:]) - b1) ** 2))) / float(b1.shape[0])
        err2[i] = np.sqrt(np.sum(np.abs(D.dot(theta[i,:])) ** 2))
        err3[i] = err1[i] + err2[i]
        lambda_estim[i] = theta[i,:].dot(S)
        spectrum[i,:] = b_spline_functions.dot(theta[i,:])

    return spectrum, lambda_estim

def _naive_estim(estim , design_m):
    A = np.concatenate((np.real(design_m), np.imag(design_m)))
    b = np.concatenate((np.real(estim), np.imag(estim)))
    naive_est , err = opt.nnls (A, b)
    return naive_est, err


def _compute_delta_m(supp,k):
    diag_A= np.concatenate(([-3*k/float(supp)],[-3*k/float(2*supp)],-k/float(supp)*np.ones(k-1),
                            [-3*k/float(2*supp)],[-3*k/float(supp)]))
    diag_B= np.concatenate(([-2*k/float(supp)],-k/float(supp)*np.ones(k-1),[-2*k/float(supp)]))
    A = np.zeros((k+2, k+3), float)
    B = np.zeros((k+1, k+2), float)
    for i in range(A.shape[0]):
        A[i,i]=diag_A[i]
        A[i,i+1]=-diag_A[i]
    for i in range(B.shape[0]):
        B[i,i]=diag_B[i]
        B[i,i+1]=-diag_B[i]

    return B.dot(A)
                  

def _computation_covariance_matrix(kappa, design_matrix, naive_estim):
    E = np.zeros((kappa.shape[0], design_matrix.shape[0]), dtype=complex)
    for i in range(kappa.shape[0]):
        E[i,:] = kappa[i,:] - design_matrix.dot(naive_estim)
        
    E = E - np.mean(E, axis=0)
    rE = np.real(E)
    iE = np.imag(E)
    
    cov_estim_rr = np.transpose(rE).dot(rE) / float(kappa.shape[0])
    cov_estim_ii = np.transpose(iE).dot(iE) / float(kappa.shape[0])
    cov_estim_ri = np.transpose(rE).dot(iE) / float(kappa.shape[0])
    cov_estim = np.bmat([[cov_estim_rr, cov_estim_ri], [cov_estim_ri.T, cov_estim_ii]])
    
    return cov_estim


def _compute_smoothed_2nd_cf(signal):
    """Function that computes several (2**7) estimators of the ratio phi'/phi where phi 
    represents the characteristic function of the shot-noise marginal.
    =====
    Args:
    =====
    - signal : a low frequency sample of the shot-noise process (np.array)

    ========
    Outputs:
    ========
    - eval_grid : evaluation points of the second characteristic function
    - kappa : several estimators of the second characteristic at each point
              in eval_grid
    - Kappa : mean of the estimators kappa
    """
    length = signal.shape[0]
    n_cuts = 2**7
    sig_length = int(length/n_cuts)
    new_length =  int(n_cuts*sig_length)
    n_cuts = int(n_cuts)
    signal = signal[0:new_length].reshape(n_cuts,sig_length)
    print "New length of the shot-noise process : {}".format(new_length)
    print "Sampled signal cut into {} subsamples of length : {} ".format(n_cuts,sig_length)

    # Computation of the ratio phi'/phi where phi is the empirical characteristic function 
    # of the marginal of the shot noise proces
    hist,bins_shot = np.histogram(signal, range=(np.nanmin(signal), np.nanmax(signal)), bins=1000, density=True)
    bins_shot2 = bins_shot[:-1] + .5 * (bins_shot[1] - bins_shot[0])
    N = 2 ** 12
    xfft = np.fft.fft(hist, N)
    dhist = np.array([a * b for a, b in zip(hist, bins_shot2)])
    dxfft = - 1j * np.fft.fft(dhist, N)
    Kappa = np.conjugate(np.divide(dxfft, xfft))
    grid_fft = np.linspace(0, N-1, N) * 2 * math.pi / float(N * (bins_shot[1] - bins_shot[0]))
    N2 = 2 ** 9
    eval_grid = grid_fft[0: N2]
    Kappa = Kappa[0: N2]
    print "Length of the shot histogram : {}".format((bins_shot[1] - bins_shot[0]))
    print "Number of evaluation points : {}".format(N2)
    print "Interval length between two observation points : {}".format(2 * math.pi / (N *( bins_shot[1] - bins_shot[0])))
    print "Maximal evaluation point : {}".format(2 * N2 * math.pi / (N * (bins_shot[1] - bins_shot[0])))

    # Computation of the ECF for each subsample
    kappa = np.zeros((signal.shape[0],N),dtype=complex)
    for i in range(signal.shape[0]):
        hist, _ = np.histogram(signal[i,:], range=(np.nanmin(signal[i,:]), np.nanmax(signal[i,:])),
                               bins=bins_shot, density=True)
        xfft = np.fft.fft(hist, N)
        dhist = np.array([a * b for a, b in zip(hist, bins_shot2)])
        dxfft = -1j * np.fft.fft(dhist,N)
        kappa[i,:] = np.conjugate(np.divide(dxfft, xfft))
    kappa= kappa[:,0:N2]

    return Kappa, kappa, eval_grid


def _pack_b_splines_and_design_matrix(impulse_response, eval_grid):
    # b) Computation of the design_matrix
    # b) 1) Computation of the b-splines basis
    C, splines, grid_x, grid_t, step_f = _compute_b_spline_basis(impulse_response, eval_grid.shape[0])
    # b) 2) Computation of the design matrix
    dM3 = _design_matrix_fortran(grid_x, grid_t, eval_grid, step_f)
    design_matrix = np.conj(dM3.dot(splines))
    A = _compute_delta_m(grid_x[-1],eval_grid.shape[0])

    return A, C, design_matrix, grid_x, splines, step_f


def shotnoise_spline_estimation(signal, impulse_response,
                                penalties = list(np.logspace(-8,8,num=80))):
    """Function which produce an estimator based on the spline method described in the paper
    "Nonparametric estimation of a shot-noise process"
    ====
    Args
    ====
    - signal : sample of a low frequency shot-noise process
    - impulse_response : the impulse response function. For example, it can be defined by:

                                def h(x):
                                    return 10*x**2*np.exp(-10*x)*(x>=0)

    - knots : the number of knots chosen for the B-splines basis
    - deg : the degree chosen for the B-splines basis
    - penalty : TODO
    - cv : TODO
    - bandwidth : TODO 

    """
    # a) Computation of the smoothed empirical second characteristic function
    Kappa, kappa, eval_grid = _compute_smoothed_2nd_cf(signal)
    # b) Computation of the design_matrix
    A, C, design_matrix, grid_x, b_spline_functions, step_f = _pack_b_splines_and_design_matrix(impulse_response, eval_grid)
    # c) Estimation procedure 
    # c) 1) Computation of a naive estimator
    naive_estim , err = _naive_estim(Kappa, design_matrix)
    # c) 2) Computation of the covariance matrix
    cov_estim = _computation_covariance_matrix(kappa, design_matrix, naive_estim)
    #c) 3) COmputation of several estimators
    estimators, lambda_estimators = _estimation(kappa, design_matrix, cov_estim, penalties, A, C, grid_x, b_spline_functions)

    return estimators, grid_x, naive_estim


def shotnoise_traditional_estimation(signal):
    """This function implements the estimation tehcnique based on the local maxima of the shot noise process.
    In a first time, it retrieves all the local maxima of the signal.
    Then, it computes an histogram of these maxima which provides an estimate of a dilated version of the marks' density
    whenever the intensity is not high enough.
    """
    ind = argrelextrema(signal, np.greater)
    ind =np.array(ind)
    ind = ind[0,:]
    y = np.array([signal[i] for i in ind])

    return y

