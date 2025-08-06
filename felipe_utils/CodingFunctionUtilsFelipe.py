'''
	Base class for temporal coding schemes
'''
## Standard Library Imports

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.CodingFunctionsFelipe import *
import numpy as np

def ApplyKPhaseShifts(x, shifts):
    K = 0
    if (type(shifts) == np.ndarray):
        K = shifts.size
    elif (type(shifts) == list):
        K = len(shifts)
    else:
        K = 1
    for i in range(0, K):
        x[:, i] = np.roll(x[:, i], int(round(shifts[i])))

    return x

def ScaleAreaUnderCurve(x, dx=0., desiredArea=1.):
    """ScaleAreaUnderCurve: Scale the area under the curve x to some desired area.

    Args:
        x (TYPE): Discrete set of points that lie on the curve. Numpy vector
        dx (float): delta x. Set to 1/length of x by default.
        desiredArea (float): Desired area under the curve.

    Returns:
        numpy.ndarray: Scaled vector x with new area.
    """
    #### Validate Input
    # assert(UtilsTesting.IsVector(x)),'Input Error - ScaleAreaUnderCurve: x should be a vector.'
    #### Calculate some parameters
    N = x.size
    #### Set default value for dc
    if (dx == 0): dx = 1. / float(N)
    #### Calculate new area
    oldArea = np.sum(x) * dx
    y = x * desiredArea / oldArea
    #### Return scaled vector
    return y


def ScaleMod(ModFs, tau=1., pAveSource=1., dt=None):
    """ScaleMod: Scale modulation appropriately given the beta of the scene point, the average
    source power and the repetition frequency.

    Args:
        ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
        tau (float): Repetition frequency of ModFs
        pAveSource (float): Average power emitted by the source
        beta (float): Average reflectivity of scene point

    Returns:
        np.array: ModFs
    """
    (N, K) = ModFs.shape
    if (dt is None): dt = tau / float(N)
    eTotal = tau * pAveSource  # Total Energy
    for i in range(0, K):
        ModFs[:, i] = ScaleAreaUnderCurve(x=ModFs[:, i], dx=dt, desiredArea=eTotal)

    return ModFs
