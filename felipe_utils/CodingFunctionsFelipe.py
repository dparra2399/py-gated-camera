#### Python imports
import math

#### Library imports
import numpy as np
from scipy import signal
# from IPython.core import debugger
# breakpoint = debugger.set_trace
import matplotlib as mpl
import matplotlib.pyplot as plt

#### Local imports

TotalEnergyDefault = 1.
TauDefault = 1.
AveragePowerDefault = TotalEnergyDefault / TauDefault

def GetCosCos(N=1000, K=4):
	"""GetCosCos: Get modulation and demodulation functions for sinusoid coding scheme. The shift
	between each demod function is 2*pi/k where k can be [3,4,5...]
	
	Args:
		N (int): N - Number of Samples
		k (int): k - Number of coding function
		freqFactor (float): Multiplicative factor to the fundamental frequency we want to use.

	Returns:
		np.array: modFs 
		np.array: demodFs 
	"""
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	t = np.linspace(0, 2*np.pi, N)
	dt = float(TauDefault) / float(N)
	#### Declare base sin function
	cosF = (0.5*np.cos(t)) + 0.5
	#### Set each mod/demod pair to its base function and scale modulations
	for i in range(0,K):
		## No need to apply phase shift to modF
		modFs[:,i] = cosF
		## Apply phase shift to demodF
		demodFs[:,i] = cosF
	#### Apply phase shifts to demodF
	shifts = np.arange(0, K)*(float(N)/float(K))
	demodFs = ApplyKPhaseShifts(demodFs, shifts)
	#### Return coding scheme
	return (modFs, demodFs)

def GetSqSq(N=1000, K=4):
	"""GetSqSq: Get modulation and demodulation functions for square coding scheme. The shift
	between each demod function is 2*pi/k where k can be [3,4,5...]. 
	
	Args:
	    N (int): Number of discrete points in the scheme
	    k (int): Number of mod/demod function pairs
	    0.5
	
	Returns:
	    np.array: modFs 
	    np.array: demodFs 
	"""
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	t = np.linspace(0, 2*np.pi, N)
	dt = float(TauDefault) / float(N)
	#### Declare base sin function
	sqF = (0.5*signal.square(t, duty=0.5)) + 0.5
	#### Set each mod/demod pair to its base function and scale modulations
	for i in range(0,K):
		## No need to apply phase shift to modF
		modFs[:,i] = sqF
		## Scale  modF so that area matches the total energy
		modFs[:,i] = ScaleAreaUnderCurve(modFs[:, i], dx=dt, desiredArea=TotalEnergyDefault)
		## Apply phase shift to demodF
		demodFs[:,i] = sqF
	#### Apply phase shifts to demodF
	shifts = np.arange(0, K)*(float(N)/float(K))
	demodFs = ApplyKPhaseShifts(demodFs, shifts)
	#### Return coding scheme
	return (modFs, demodFs)

def GetHamK3(N=1000, modDuty = 1./ 6.):
	"""GetHamK3: Get modulation and demodulation functions for the coding scheme
		HamK3 - Sq16Sq50.	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 3
	dt = float(TauDefault) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	for i in range(0,K):
		modFs[0:math.floor(modDuty*N),i] = AveragePowerDefault
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty = 1./2.
	for i in range(0,K):
		demodFs[0:math.floor(demodDuty*N),i] = 1.
	## Apply necessary phase shift
	shifts = [0, (1./3.)*N, (2./3.)*N]
	demodFs = ApplyKPhaseShifts(demodFs, shifts)

	return (modFs, demodFs)



def GetHamK4(N=1000, modDuty = 1. / 12.):
	"""GetHamK4: Get modulation and demodulation functions for the coding scheme HamK4	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 4
	dt = float(TauDefault) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	for i in range(0, K):
		modFs[0:math.floor(modDuty * N), i] =  AveragePowerDefault
		# if i > 1:
		# 	modFs[:, i] = np.tile(modFs[:int(N // 2), i], 2)
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty1 = np.array([6./12.,6./12.])
	shift1 = 5./12.
	demodDuty2 = np.array([6./12.,6./12.])
	shift2 = 2./12.
	demodDuty3 = np.array([3./12.,4./12.,3./12.,2./12.])
	shift3 = 0./12.
	demodDuty4 = np.array([2./12.,3./12,4./12.,3./12.])
	shift4 = 4./12.
	shifts = [shift1*N, shift2*N, shift3*N, shift4*N]
	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4]
	for i in range(0,K):
		demodDuty = demodDutys[i]
		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
		for j in range(len(demodDuty)):
			if((j%2) == 0):
				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.
	## Apply necessary phase shift
	demodFs = ApplyKPhaseShifts(demodFs, shifts)

	return (modFs, demodFs)


def GetHamK5(N=1000, modDuty = 1. / 30.):
	"""GetHamK5: Get modulation and demodulation functions for the coding scheme HamK5.	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 5
	dt = float(TauDefault) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	for i in range(0,K):
		modFs[0:math.floor(modDuty*N),i] = AveragePowerDefault
		# if i == 2:
		# 	modFs[:, i] = np.tile(modFs[:, i], 2)[0::2]
		# elif i == 3:
		# 	modFs[:, i] = np.tile(modFs[:, i], 4)[0::4]
		# elif i == 4:
		# 	modFs[:, i] = np.tile(modFs[:, i], 7)[0::7]
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty1 = np.array([15./30.,15./30.])
	shift1 = 15./30.
	demodDuty2 = np.array([15./30.,15./30.])
	shift2 = 7./30.
	demodDuty3 = np.array([8./30.,8./30.,7./30.,7./30.])
	shift3 = 3./30.
	demodDuty4 = np.array([4./30.,4./30.,4./30.,4./30.,3./30.,4./30.,4./30.,3./30.])
	shift4 = 1./30.
	demodDuty5 = np.array([2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,
							3./30.,2./30.,2./30.,2./30.,2./30.,3./30.,2./30])
	shift5 = 4./30.
	shifts = [shift1*N, shift2*N, shift3*N, shift4*N, shift5*N]
	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4, demodDuty5]
	for i in range(0,K):
		demodDuty = demodDutys[i]
		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
		for j in range(len(demodDuty)):
			if((j%2) == 0):
				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.

	## Apply necessary phase shift
	demodFs = ApplyKPhaseShifts(demodFs, shifts)

	return (modFs, demodFs)

def GetMultiFreqCosK5(N=1000, highFreqFactor=7.):
	"""GetMultiFreqCos: Returns a coding scheme based on square waves that goes the following way. 
	Let w be the repetition frequency. The first code is a SqCode with rep freq w. The second
	code is a Sq Code with rep freq 2*w. The Kth code is a SqCode with a rep freq K*w 
	
	Args:
	    N (int): Number of discrete points in the scheme
	    k (int): Number of mod/demod function pairs	
	Returns:
	    np.array: modFs 
	    np.array: demodFs 
	"""
	K=5
	ModFs = np.zeros((N,K))
	DemodFs = np.zeros((N,K))
	t = np.linspace(0, 2*np.pi, N)
	HighFreqCosF = (0.5*np.cos(highFreqFactor*t)) + 0.5 # Phase = 0
	HighFreqCosF90 = np.roll(HighFreqCosF, int(round((N/4) / highFreqFactor))) # Phase 90 degrees
	#### Generate the fundamental frequency sinusoids
	(CosModFs,CosDemodFs) = GetCosCos(N=N,K=3)
	ModFs[:,0:3] = CosModFs
	DemodFs[:,0:3] = CosDemodFs
	#### Generate the high frequency sinusoid functions
	ModFs[:,3] = HighFreqCosF*2.	
	DemodFs[:,3] = HighFreqCosF
	ModFs[:,4] = ModFs[:,3]
	DemodFs[:,4] = HighFreqCosF90

	return (ModFs,DemodFs)


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