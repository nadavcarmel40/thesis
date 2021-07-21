import numpy as np
#################################################################################
##############                       constants                   ################
#################################################################################

h=6.626e-34
hbar=1.05e-34
freq=6e9

sigmaForError = np.pi/100
meanForError = 0

decoherence_mode = 0
num_counting = 3 #number of counting qubits
num_measuredOp = 1
beta = 0.02