# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
#from cubical.machines.parameterised_machine import ParameterisedGains
from __future__ import print_function
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from numpy.ma import masked_array
import cubical.kernels

from cubical.tools import logger
log = logger.getLogger("solver") #pphase

def _normalise(x, dtype):
    """
    Helper function: normalises array to [0,1] interval.

    """

    if len(x) > 1:
        return (x/min(x)).astype(dtype)
    elif len(x) == 1:
        return np.ones(1, dtype)
    else:
        return x

def _normalise_lm(sources):
    """
    Helper function: normalizes (sources) array to [0,1] deg range.
    
    Since I am having very small l and m values with respect to a 1 deg field of view,
    I want to normalise the l and m coordinates such that they do appear bigger and scale up
    to a common sky at the end of the day.

    """

    l = sources[:, 1]
    m = sources[:, 2]

    min_l = np.min(l)
    min_m = np.min(m)
    max_l = np.max(l)
    max_m = np.max(m)

    diag_length = np.sqrt((max_l-min_l)**2 + (max_m-min_m)**2)

    normalised_l = (l-min_l)/diag_length
    normalised_m = (m-min_m)/diag_length

    ##Keep same phase centre or not?
    #normalised_l -= normalised_l[0]
    #normalised_m -= normalised_m[0]
    
    sources[:, 1] = normalised_l
    sources[:, 2] = normalised_m

    return sources

def _new_normalise(sources):
    """
    I want to make a new normalisation function to make l and m fit in [-1, 1].
    
    """
    
    l = sources[:, 1]
    m = sources[:, 2]

    min_l = min(np.abs(l))
    min_m = min(np.abs(m))
    max_l = max(np.abs(l))
    max_m = max(np.abs(m))
    
    if min_l < min_m:
        mini = min_l
    else:
        mini = min_m
        
    if max_l < max_m:
        maxi = max_m
    else:
        maxi = max_l
    
    normalised_l = (l - mini)/(maxi - mini)
    normalised_m = (m - mini)/(maxi - mini)

    sources[:, 1] = normalised_l
    sources[:, 2] = normalised_m

    return sources

def _make_basis_vec(n_param, l_s, m_s):
    """
    Generating the basis polynomial to compute the phase equation. Right now, 
    it is of the form [1, l_s, m_s, l_s**2, m_s**2, ...] with l_s and m_s being 
    scalars. The function returns a vector of length n_params.

    """
    
    N = (n_param+1) // 2
    lvec = (np.tile(l_s, N-1) ** np.arange(1, N))
    mvec = (np.tile(m_s, N-1) ** np.arange(1, N))

    main_vec = np.ones((N-1, 2))
    main_vec[:, 0] = lvec
    main_vec[:, 1] = mvec
    main_vec = (main_vec).flatten()
    
    return np.insert(main_vec, 0, [1])

def _get_basis(n_param, sources):
    """
    Get basis matrix of shape (n_params, n_dir). Both l and m, which represent the
    direction cosine coordinates of the sources are each vectors of length n_dir.
    
    """

    #Get the dimension and the direction cosines.
    l = sources[:, 1]
    m = sources[:, 2]

    #Get "n_dir" dimension.
    n_dir = sources.shape[0] #len(l)
    basis = np.zeros((n_param, n_dir))

    for s in range(n_dir):
        basis[:, s] = _make_basis_vec(n_param, l[s], m[s])

    return basis

import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

#*************************************************************************************************
#sources = np.array([[1, 0, 0], [1, np.deg2rad(0.77), np.deg2rad(1)], 
#                    [1, np.deg2rad(-0.3), np.deg2rad(-0.6)], [1, np.deg2rad(-0.4), np.deg2rad(0.5)], 
#                    [1, np.deg2rad(0.5), np.deg2rad(-0.5)]])

sources = np.array([[  1.00000000e+00,   3.83510434e-16,   0.00000000e+00],
       [  1.00000000e+00,  -1.34390352e-02,   1.74532925e-02],
       [  1.00000000e+00,   5.23598776e-03,  -1.04719755e-02],
       [  1.00000000e+00,   6.98131701e-03,   8.72664626e-03],
       [  1.00000000e+00,  -8.72664626e-03,  -8.72664626e-03]])
#sources = _normalise_lm(sources)
sources = _new_normalise(sources)

#*************************************************************************************************

class ParametrisedPhaseMachine(PerIntervalGains):
    """
    This class implements the phase-only parametrised gain machine (diag-diag).

    """

    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a phase-only parametrised gain machine.
        
        **Must modify the arguments!**

        Args:
            label (str):
                Label identifying the Jones term.
            data_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
            ndir (int):
                Number of directions.
            nmod (nmod):
                Number of models.
            chunk_ts (np.ndarray):
                Times for the data being processed.
            chunk_fs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options.

        """
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options) 

        #import pdb; pdb.set_trace()

        ##options will contain the extra material I need such as n_param or l and m.
        self.n_param = options.get("pphase-nparam", 3)
        self.jhj_diag = options.get("pphase-diag-jhj", True)
        #print("{} nparam".format(self.n_param)>>log(2))
        #self.l = options["pphase-l"]
        #self.m = options["pphase-m"]
        self.l = sources[:, 1]
        self.m = sources[:, 2]
        self.n_dir = sources.shape[0]

        ##Initial guess alpha0 (use alpha for convenience sake).
        np.random.seed(3)
        #self.alpha = 0.05*np.random.randn(self.n_ant, self.n_param, self.n_cor)
        self.alpha = np.zeros((self.n_ant, self.n_param, self.n_cor))
        #self.alpha[:, 0, :] = 1

        ##I am making basis an attribute so that it is easier to compute gains
        ##inside implement_update().
        self.basis = _get_basis(self.n_param, sources)

        self.chunk_fs = _normalise(chunk_fs, self.ftype)

        self.make_gains()

        self._labels = dict(phase=0, param=1)
        self.param_shape = [self.n_timint, self.n_freint, 
                            self.n_ant, self.n_param, self.n_cor, self.n_cor]

        self.posterior_alpha_error = None

        self.residuals = np.empty_like(data_arr)
        
    def make_gains(self):
        """
        Returns the gains using alpha (array containing gain parameters per antenna
        per correlation).

        """ 

        for s in range(self.n_dir):
            for t in range(self.n_timint):
                for f in range(self.n_fre):
                    for p in range(self.n_ant):
                        #To correct for the dimension issue ((n_param,) instead of (n_param, 1)).
                        alpha_vec0 = (self.alpha[p, :, 0]).reshape(self.n_param)
                        alpha_vec1 = (self.alpha[p, :, 1]).reshape(self.n_param)
                        #phase_equation = np.dot(alpha_vec, basis[:, s])
                        #We want to vary the gains with frequency, and thus, indexing within n_fre instead of n_freint.
                        self.gains[s, t, f, p, 0, 0] = np.exp(1.0j * np.dot(alpha_vec0, self.basis[:, s])/self.chunk_fs[f])
                        self.gains[s, t, f, p, 1, 1] = np.exp(1.0j * np.dot(alpha_vec1, self.basis[:, s])/self.chunk_fs[f])

    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        return False

    @property 
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        ##Assuming diagonal gains!
        return self.n_param*2
    
    def compute_jacobian_residual(self, data_arr, model_arr, gains):
        """
        Returns the Jacobian and residual.
    
        Args:
            data_arr (np.array):
                Array containing the observed visibilities.
            model_arr (np.array):
                Array containing the model visibilities.
            gains (np.array):
                Array containing the current gain estimates.
        
        Returns:
            jac (np.array):
                Array of shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_ant, n_param, n_cor)
                containing the Jacobian.
            residual (np.array):
                Array containing the residual visibilities.
            
        """

        #Initialise Jacobian.
        self.jac_shape = [self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_ant, self.n_param, self.n_cor] 
        jac = np.zeros(self.jac_shape, dtype=self.dtype)

        #----using cubical-----------#
        ##Need to check this with Jonathan!
        #self.residuals = self.compute_residual(data_arr, model_arr, self.residuals)

        #Initialise residual as data since we just need to subtract the model in every direction.
        residual = data_arr.copy()

        for t in range(self.n_tim):
            tt = t//self.t_int
            for f in range(self.n_fre):
                for p in range(self.n_ant):
                    for q in range(p):  #note only doing this for q < p
                        for s in range(self.n_dir):
                            #Subtract model for each direction.
                            residual[0, t, f, p, q] -= gains[s, tt, f, p] * model_arr[s, 0, t, f, p, q] * np.conj(gains[s, tt, f, q].T)
                            for k in range(self.n_cor):
                                #Get Jacobian.
                                for param in range(self.n_param):
                                    #Get partial derivative of the phase.
                                    dphidalpha = 1.0j*self.basis[param, s]/self.chunk_fs[f]
                                    jac[t, f, p, q, k, p, param, k] += dphidalpha * gains[s, tt, f, p, k, k] * model_arr[s, 0, t, f, p, q, k, k] * np.conj(gains[s, tt, f, q, k, k]) #I do not need to transpose gains_q (scalar).
                                    jac[t, f, p, q, k, q, param, k] += -dphidalpha * gains[s, tt, f, p, k, k] * model_arr[s, 0, t, f, p, q, k, k] * np.conj(gains[s, tt, f, q, k, k])

                        #Set [q,p] element as conjugate of [p,q] (LB - is this correct for the Jacobian as well?)
                        residual[0, t, f, q, p] = np.conj(residual[0, t, f, p, q])
                        jac[t, f, q, p] = np.conj(jac[t, f, p, q])
    
        ##Reshape the Jacobian to a 2D shape and residuals to 1D.
        jac = np.reshape(jac, (self.n_tim*self.n_fre*self.n_ant*self.n_ant*self.n_cor, self.n_ant*self.n_param*self.n_cor))
        #self.residual = np.reshape(self.residual, (self.n_tim*self.n_fre*self.n_ant*self.n_ant*self.n_cor*self.n_cor))

        return jac, residual #self.residuals

    def get_xx_yy_residual(self, residual):
        """
        This function extracts the XX and YY components only from the residuals and
        returns the new residuals.
    
        Args:
            residual (np.array):
                Array of shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
                containing the residual visibilities. The n_mod dimension has not 
                been considered.
        
        Returns:
            new_residual (np.array):
                Array of shape (n_mod*n_tim*n_fre*n_ant*n_ant*n_cor) containing 
                the residual visibilities for only XX and YY terms.

        """

        ##Initialise new_residual since we are dealing with diagonal data and needs
        ##to match with the shape of the Jacobian
        new_residual = np.zeros((self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor), dtype=self.dtype)

        for k in range(self.n_cor):
            new_residual[..., k] = residual[..., k, k]

        return new_residual.reshape(self.n_tim*self.n_fre*self.n_ant*self.n_ant*self.n_cor)

    def compute_blockwise_jhj(self, jac):
        """
        This function is relevant when one wants to compute the update using only
        the block diagonal of the JHJ. For now, the whole JHJ is being stored despite
        the zero entries.

        Args:
            jac (np.array):
                Array of shape (n_tim*n_fre*n_ant*n_ant*n_cor, n_ant*n_param*n_cor)
                containing the Jacobian.
        
        Returns:
            jhj (np.array):
                Array of shape (n_ant*n_param*n_cor, n_ant*n_param*n_cor) containing 
                JHJ.

        """

        ##Get the conjugate transpose of the Jacobian.
        jh = np.conjugate(jac.T)

        ##Initialise jhj.
        jhj = np.zeros((self.n_ant*self.n_param*self.n_cor, self.n_ant*self.n_param*self.n_cor), dtype=self.dtype)

        for k in range(self.n_ant):
            jhj[k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor, k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor] = np.dot(jh[k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor, :], jac[:, k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor])
        
        return jhj

    def compute_js(self, data_arr, model_arr):
        """
        This function computes the components JHJ and JHr for the update rule.
        JHJ can be computed either entirely or block diagonally.

        Args:
            data_arr (np.array):
                Array containing observed visibilities.
            model_arr (np.array):
                Array containing model visibilities.

        
        Returns:
            jhj (np.array):
                Array of shape (n_ant*n_param*n_cor, n_ant*n_param*n_cor) containing 
                JHJ.
            jhr (np.array):
                Array of shape (n_ant*n_param*n_cor) containing JHr.

        """

        #import pdb; pdb.set_trace()

        jac, residual_2x2 = self.compute_jacobian_residual(data_arr, model_arr, self.gains)
        #Change shape of residual to match the second axis of JH.
        residual = self.get_xx_yy_residual(residual_2x2)

        jh = np.conjugate(jac.T)

        if self.jhj_diag: #solve == "block_diag_jhj":
            jhj = self.compute_blockwise_jhj(jac)

        else: #using full JHJ.
            jhj = np.dot(jh, jac)

        ##Initialise JHr.
        jhr = np.zeros((self.n_ant*self.n_param*self.n_cor), dtype=self.dtype)
        jhr = np.dot(jh, residual)

        return jhj, jhr

    def implement_update(self, jhr, jhj):
        """
        Internal method implementing a parameter update. The standard compute_update() implementation 
        calls compute_jacobian_residual() and _implement_update() at each iteration step.

        Args:
            jhr (np.array):
                Array containing JHr.
            jhj (np.array):
                Array containing model JHJ.

        
        Updates:
            alpha (np.array):
                Array containing current gain parameters.
            gains (np.array):
                Array containing current gain estimates computed using updated alpha.

        """

        #import pdb; pdb.set_trace()

        if self.jhj_diag:
            ##Initialise delta_alpha = change in alpha.
            delta_alpha = np.zeros((self.n_ant, self.n_param*self.n_cor), dtype=self.alpha.dtype)

            for k in range(self.n_ant):
                ##Each block_jhj has shape (n_param*n_cor, n_param*n_cor) contains 
                ##the derivatives of data_arr w.r.t. alpha @ant k. 
                block_jhj = jhj[k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor, k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor]
                #And thus, delta_alpha is computed for each antenna at a time.
                delta_alpha[k] = (np.linalg.solve(block_jhj, jhr[k*self.n_param*self.n_cor:(k+1)*self.n_param*self.n_cor])).real
        
        else:
            #In the case of full JHJ, delta_alpha gets computed at once unlike above.
            delta_alpha = np.linalg.solve(jhj, jhr)
            #alpha is real and it is required to use real delta_alpha.
            delta_alpha = np.real(delta_alpha)

        delta_alpha = np.reshape(delta_alpha, (self.n_ant, self.n_param, self.n_cor)) 
        self.alpha += 0.5*delta_alpha

        #Need to turn updated parameters into gains.
        self.make_gains()

        np.save("alpha.npy", self.alpha)
        np.save("gains_lm.npy", self.gains)

        #chisq_num = 2*self.residual.size - self.alpha.size
        #chi2 = (np.linalg.norm(self.residual)) / chisq_num
        #log(2).print("chi2 = {}".format(chi2))

    def compute_update(self, model_arr, data_arr):
        """
        This function calls implement_update() and computes the update rule.

        Args:
            model_arr (np.array):
                Array containing model visibilities.
            data_arr (np.array):
                Array containing observed visibilities. 
    
        """

        #import pdb; pdb.set_trace()
        
        #Get JHJ and JHr to compute the update rule.
        jhj, jhr = self.compute_js(data_arr, model_arr)

        ##I want to save Re(JHJ).
        np.save("re_jhj.npy", jhj.real)

        #Updates the gain parameters and also computes the new gains.
        self.implement_update(jhr, jhj)

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        exportables = PerIntervalGains.exportable_solutions()

        exportables.update({
            "params": (0., ("nparams", "time", "freq", "ant", "corr")),
            "params.err": (0., ("time", "freq", "ant", "params")),
        })
        
        return exportables

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """

        # import pdb; pdb.set_trace()
        solutions = PerIntervalGains.export_solutions(self)

        alpha_store = self.alpha[np.newaxis, np.newaxis, :, :]

        for label, num in self._labels.items():
            solutions[label] = masked_array(alpha_store), self.interval_grid
            if self.posterior_alpha_error is not None:
                solutions[label+".err"] = masked_array(np.zeros_like(alpha_stored)), self.interval_grid

        return solutions
