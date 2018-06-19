"""Inverse transform sampling, for sampling from arbitrary probability density
functions."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from numpy.random import uniform

from numpy.polynomial.chebyshev import chebfit, chebval, chebint

# Functions to check if a numerical quadrature is convergent

_MESSAGE = "The integral is probably divergent, or slowly convergent."

def _convergent(quadrature):
    if len(quadrature)>3 and quadrature[3] == _MESSAGE:
        return False
    else:
        return True


def normalize(pdf, lower_bd=-np.inf, upper_bd=np.inf, vectorize=False):
    """Normalize a non-normalized PDF.
    
    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    vectorize: boolean
        Vectorize the function. This slows down function calls, and so is
        generally set to False.
    
    Returns
    -------
    pdf_norm : function
        Function with same signature as pdf, but normalized so that the integral
        between lower_bd and upper_bd is close to 1. Maps nicely over iterables.
    """
    if lower_bd >= upper_bd:
        raise ValueError('Lower bound must be less than upper bound.')
    quadrature = quad(pdf, lower_bd, upper_bd,full_output=1)
    if not _convergent(quadrature):
        raise ValueError('PDF integral likely divergent.')
    A = quadrature[0]
    pdf_normed = lambda x: pdf(x)/A if lower_bd <= x <= upper_bd else 0
    if vectorize:
        def pdf_vectorized(x):
            try:
                return pdf_normed(x)
            except ValueError:
                return np.array([pdf_normed(xi) for xi in x])
        return pdf_vectorized
    else:
        return pdf_normed


def get_cdf(pdf, lower_bd=-np.inf, upper_bd=np.inf):
    """Generate a CDF from a (possibly not normalized) pdf.
    
    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    
    Returns
    -------
    cdf : function
        The cumulative density function of the (normalized version of the)
        provided pdf. Will return a float if provided with a float or int; will
        return a numpy array if provided with an iterable.

    """
    pdf_norm = normalize(pdf, lower_bd, upper_bd)
    
    def cdf_number(x):
        "Numerical cdf"""
        if x < lower_bd:
            return 0.0
        elif x > upper_bd:
            return 1.0
        else:
            return quad(pdf_norm,lower_bd,x)[0]

    def cdf_vector(x):
        try:
            return np.array([cdf_number(xi) for xi in x])
        except AttributeError:
            return cdf_number(x)

    return cdf_vector

############################################
### CHEBYSHEV APPROXIMATION OF PDF & CDF ###
############################################

# This section follows the work of Olver & Townsend (2013), who suggest using
# Chebyshev polynomials to approximate the PDF. This allows for trivial
# construction of the CDF, as these polynomials can be integrated in closed
# form.


def _chebnodes(a,b,n):
    """Chebyshev nodes of rank n on interal [a,b]."""
    if not a < b:
        raise ValueError('Lower bound must be less than upper bound.')
    return np.array([1/2*((a+b)+(b-a)*np.cos((2*k-1)*np.pi/(2*n))) for k in range(1,n+1)])


def adaptive_chebfit(pdf, lower_bd, upper_bd, eps=10**(-15)):
    """Fit a chebyshev polynomial, increasing sampling rate until coefficient
    tail falls below provided tolerance.
    
    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    eps: float
        Error tolerance of Chebyshev polynomial fit of PDF.

    Returns
    -------
    x : array
        The nodes at which the polynomial interpolation takes place. These are
        adaptively chosen based on the provided tolerance.
    coeffs : array
        Coefficients in Chebyshev approximation of the PDF.

    Notes
    -----
    This fit defines the "error" as the magnitude of the tail of the Chebyshev
    coefficients. Computing the true error (i.e. discrepancy between the PDF and
    it's approximant) would be much slower, so we avoid it and use this rough
    approximation in its place.

    """
    i = 4
    error = eps+1 # so that it runs the first time through
    while error > eps:
        n = 2**i + 1
        x = _chebnodes(lower_bd,upper_bd,n)
        y = pdf(x)
        coeffs = chebfit(x,y,n-1)
        error = max(np.abs(coeffs[-5:]))
        i += 1
    return x,coeffs
    

def chebcdf(pdf, lower_bd, upper_bd, eps=10**(-15)):
    """Get Chebyshev approximation of the CDF.
    
    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    eps: float
        Error tolerance of Chebyshev polynomial fit of PDF.
    
    Returns
    -------
    cdf : function
        The cumulative density function of the (normalized version of the)
        provided pdf. The function cdf() takes an iterable of floats or doubles
        as an argument, and returns an iterable of floats of the same length.
    """
    if not (np.isfinite(lower_bd) and np.isfinite(upper_bd)):
        raise ValueError('Bounds must be finite.')
    if not lower_bd < upper_bd:
        raise ValueError('Lower bound must be less than upper bound.')

    x,coeffs = adaptive_chebfit(pdf, lower_bd, upper_bd, eps)
    int_coeffs = chebint(coeffs)
    # offset and scale so that it goes from 0 to 1, i.e. is a true CDF.
    offset = chebval(lower_bd, int_coeffs)
    scale = chebval(upper_bd, int_coeffs) - chebval(lower_bd, int_coeffs)
    cdf = lambda x: (chebval(x,int_coeffs) - offset) / scale
    return cdf

# Sample via root-finding on CDF

def sample(pdf, num_samples, lower_bd=-np.inf, upper_bd=np.inf, guess=0, chebyshev=False):
    """Sample from an arbitrary, unnormalized PDF.
    
    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    num_samples : int
        The number of samples to be generated.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    guess : float or int
        Initial guess for the numerical solver to use when inverting the CDF.
    chebyshev: Boolean, optional (default=False)
        If True, then the CDF is approximated using Chebyshev polynomials.
    
    Returns
    -------
    samples : numpy array
        An array of samples from the provided PDF, with support between lower_bd
        and upper_bd. 

    Notes
    -----
    For a unimodal distribution, the mode is a good choice for the parameter
    guess. Any number for which the CDF is not extremely close to 0 or 1 should
    be acceptable. If the cdf(guess) is near 1 or 0, then its derivative is near 0,
    and so the numerical root finder will be very slow to converge.

    This sampling technique is slow (~3 ms/sample for a unit normal with initial
    guess of 0), since we re-integrate to get the CDF at every iteration of the
    numerical root-finder. This is improved somewhat by using Chebyshev
    approximations of the CDF, but the sampling rate is still prohibitively slow
    for >100k samples.

    """
    if chebyshev:
        if not (np.isfinite(lower_bd) and np.isfinite(upper_bd)):
            raise ValueError('Bounds must be finite for Chebyshev approximation of CDF.')
        cdf = chebcdf(pdf, lower_bd, upper_bd)
    else:
        cdf = get_cdf(pdf, lower_bd, upper_bd)
    seeds = uniform(0, 1, num_samples)
    samples = []
    for seed in seeds:
        shifted = lambda x: cdf(x)-seed
        soln = root(shifted,guess)
        samples.append(soln.x[0])
    return np.array(samples)
