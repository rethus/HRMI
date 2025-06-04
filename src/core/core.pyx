cimport numpy as np  # import C-API
import numpy as np
from libcpp cimport bool
from utilities import logger

#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "HyLoReF.h":
    void HyLoReF(double *removedData, double *testData, double *weightData, double *weightData2, int numUser, int numService, int dim,
              double gamma, double lmda, double eta, double eta2, double theta, int maxIter, double alphaInit, double decayRate, int decaySteps,
              double *Udata, double *Sdata, double *lossData, double *errData, bool debugMode)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedMatrix, testMatrix, weightMatrix, weightMatrix2, para):
    cdef int numService = removedMatrix.shape[1]
    cdef int numUser = removedMatrix.shape[0]
    cdef int dim = para['dimension']
    cdef double gamma = para['gamma']
    cdef double lmda = para['lambda']
    cdef double eta = para['eta']
    cdef double eta2 = para['eta2']
    cdef double theta = para['theta']
    cdef int maxIter = para['maxIter']
    cdef double alphaInit = para['alphaInit']
    cdef double decayRate = para['decayRate']
    cdef int decaySteps = para['decaySteps']
    cdef bool debugMode = para['debugMode']

    # initialization
    cdef np.ndarray[double, ndim=2, mode='c'] U = np.random.normal(0, 0.1, (numUser, dim))
    cdef np.ndarray[double, ndim=2, mode='c'] S = np.random.normal(0, 0.1, (numService, dim))
    cdef np.ndarray[double, ndim=1, mode='c'] loss = np.zeros(maxIter)
    cdef np.ndarray[double, ndim=2, mode='c'] err = np.zeros((maxIter, 2))

    logger.info('Iterating...')

    # Wrap up HyLoReF.cpp
    HyLoReF(<double *> (<np.ndarray[double, ndim=2, mode='c']> removedMatrix).data,
         <double *> (<np.ndarray[double, ndim=2, mode='c']> testMatrix).data,
         <double *> (<np.ndarray[double, ndim=2, mode='c']> weightMatrix).data,
         <double *> (<np.ndarray[double, ndim=2, mode='c']> weightMatrix2).data,
         numUser,
         numService,
         dim,
         gamma,
         lmda,
         eta,
         eta2,
         theta,
         maxIter,
         alphaInit,
         decayRate,
         decaySteps,
         <double *> U.data,
         <double *> S.data,
         <double *> loss.data,
         <double *> err.data,
         debugMode
         )

    return loss, err
#########################################################
