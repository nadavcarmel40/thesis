from qutip import *
from qiskit.visualization import plot_histogram
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from qiskit.quantum_info.operators import Operator
from collections import Counter
from copy import deepcopy
from scipy.linalg import logm
from simulators.Constants import *

#################################################################################
##############                        utils                      ################
#################################################################################
def produce_start_state(state):
    """
    state = string. '+' for |+>
                    '-' for |->
                    'g' for |0>
                    'e' for |1>
    """
    g = basis(2, 0)
    e = basis(2, 1)

    if state == "+":
        state = 1 / np.sqrt(2) * (e + g)
    elif state == "-":
        state = 1 / np.sqrt(2) * (e - g)
    elif state == "g":
        state = g
    elif state == "e":
        state = e
    return state * state.dag()

def get_bloch_from_rho(rho):
    """
    (x,y,z) is point in bloch sphere.
    rho is density matrix as numpy array
    """
    r00 = rho[0, 0]
    r10 = rho[1, 0]
    r01 = rho[0, 1]
    r11 = rho[1, 1]
    return (r01 + r10, 1j * (r01 - r10), r00 - r11)

def get_rho_from_bloch(x,y,z):
    """
    (x,y,z) is point in bloch sphere.
    returns density matrix
    """
    return 0.5*qeye(2) + x/2*sigmax() + y/2*sigmay() + z/2 * sigmaz()

def QPE(N,U,measure=False, iQFT=True, projection=False, FFed=True):
    """
    :param N:  = percision level = number of counting qubits
    :param U: generator of Quantum Operator that is measured
    :param measure: bool, measure the counting qubits
    :returns commands for QPA for n+N qubits where N=# counting qubits and n = #size of U Operator to measure in qubits
    """
    # initialization part
    commands = [[('h',N-i-1) for i in range(N)]]
    # apply U part
    for i in range(N):
        commands.append([('CU '+str(i), i, FFed, U)]) # third index has no meaning
    #QFTdagger part
    if iQFT:
        commands.append([('h',N-1)])
        for i in range(1,N-1):
            commands.append([('R2inv',N-i,N-1-i)])
            commands.append([('h',N-1-i), ('R3inv',N-i, N-2-i)])
        commands.append([('R2inv',1,0)])
        commands.append([('h',0)])
    #project
    if projection:
        Gate = (-1j * U).expm()
        state = Gate.eigenstates()[1][0]  # take ground state
        stateDm = state * state.dag()
        projector = (tensor([qeye(2) if i < num_counting else stateDm for i in range(num_counting + 1)]))
        commands.append([('applyOperator',projector)])
    #measurment
    if measure:
        commands.append([('m',[i for i in range(N)])])
    return commands

def QPE1(N,U,measure=False, iQFT=True, multprojection=False):
    """
    :param N:  = percision level = number of counting qubits
    :param U: generator of Quantum Operator that is measured
    :param measure: bool, measure the counting qubits
    :param multprojection: bool, does multiple projections if true
    :returns commands for QPA for n+N qubits where N=# counting qubits and n = #size of U Operator to measure in qubits
    with post selection after each stage of applying U
    """
    # initialization part
    commands = [[('h',N-i-1) for i in range(N)]]
    Gate = (-1j * U).expm()
    state = Gate.eigenstates()[1][0]  # take ground state
    stateDm = state * state.dag()
    projector = (tensor([qeye(2) if i < num_counting else stateDm for i in range(num_counting + 1)]))
    # apply U part
    for i in range(N):
        commands.append([('CU '+str(i), i, -1, U)]) # third index has no meaning
        if multprojection:
            commands.append([('applyOperator', projector)])
    #QFTdagger part
    if iQFT:
        commands.append([('h',N-1)])
        for i in range(1,N-1):
            commands.append([('R2inv',N-i,N-1-i)])
            commands.append([('h',N-1-i), ('R3inv',N-i, N-2-i)])
        commands.append([('R2inv',1,0)])
        commands.append([('h',0)])
    #measurment
    if measure:
        commands.append([('m',[i for i in range(N)])])
    return commands

def initialize(N, U, k):
    """
    returns density matrix of |0> for counting qubits followed by |u> for other qubits
    N is number of counting qubits,
    U is the Operator to which we want to measure phase (Qobj)
    k is the index of U's eigenstate to be initialized by order from ground to most exited
    """
    state = U.eigenstates()[1][k]
    stateDm = state * state.dag()
    result = tensor([basis(2, 0) * basis(2, 0).dag() if i < N else stateDm for i in range(N + 1)])
    return result

def bin2num(bins):
    """
    bins is a string of the binary number. returns represented number
    """
    result = 0
    for i in range(len(bins)):
        result += int(bins[i])*2**(len(bins)-1-i)
    return result

def QPEforPHASE(decoherence_mode, showQubits=False, printResult = False):
    """
    :param showQubits: if true, shows qubits
    :param decoherence_mode: decoherence_mode = 0: run with no decoherence
    decoherence_mode = 1: run with random gate phase errors
    decoherence_mode = 2: run with random control qubit errors
    decoherence_mode = 3: run with 1+2
    decoherence_mode = 4: run with kraus decoherence on big time steps
    decoherence_mode = 5: run with kraus decoherence on small time steps

    :return: dict of counting qubits product space state and the probability
     to get them
    """
    global num_counting
    global rho_0
    global commands
    global qr
    # create string of measurments
    qr.run(commands)
    if showQubits:
        showQubits(qr)
    state = qr.state.ptrace([i for i in range(num_counting)])
    p = list(np.real(np.diag(state.data.toarray())))
    # find 2 highest vals
    results = {}
    for i, prob in enumerate(p):
        res="{0:b}".format(i)
        while len(res)<num_counting:
            res = '0'+res
        results[res]=prob
    k = Counter(results)
    high = k.most_common(2)
    highest = []
    for i in high:
        highest.append(bin2num(i[0]))
    try:
        if printResult:
            print("theta is between " + str(highest[0] / 2 ** num_counting) + " and " + str(highest[1] / 2 ** num_counting))
        return results
    except:
        if printResult:
            print("result is " + str(highest[0] / 2 ** num_counting))
        return results

def applyProjection(qr,U):
    Gate = (-1j*U).expm()
    state = Gate.eigenstates()[1][0] #take ground state
    stateDm = state * state.dag()
    projector = (tensor([qeye(2) if i < num_counting else stateDm for i in range(num_counting + 1)]))
    commands = [[('applyOperator',projector)]]
    qr.run(commands)
    return qr

def addNoise2G(G,beta=0, normalization=None):
    """
    :param G: generator of gate operator to add noise to - Qobj, must be in small space of only the measured qubits
    :param beta: (1-beta) = the probability that acting on groundstate gives right result
    :return: generator of gate with noise - Qobj
    """
    n = G.data.shape[0]
    nonH = Qobj(np.random.normal(size=(n,n))+1j*np.random.normal(size=(n,n)), dims=G.dims)
    H = 1/2*(nonH+nonH.dag())
    #normalize
    H = norm(np.max(G))*H
    if G.isherm:
        return G+beta*H
    else:
        return G+beta*H*1j

def showQubits(qr, color=None):
    for q in range(qr.N):
        plt.style.use('classic')
        pnt = get_bloch_from_rho(qr.state.ptrace(q))
        bloch = Bloch()
        bloch.add_points(pnt)
        if color is not None:
            bloch.sphere_color = color
        bloch.add_annotation(1 / np.sqrt(3.5) * (basis(2, 1) + 2j * basis(2, 0)), 'qubit ' + str(q))
        bloch.show()

def ShorsU(a,N):
    """
    :param a: integer from 0 to N-1 such that gcd(a,N) = 1
    :param N: integer
    :return: Qobj for U such that U|x>=|axmodN>
    """
    num_digits = len(bin(N)[2:])
    U = np.zeros((2**num_digits,2**num_digits))
    for j in range(0,2**num_digits):
        if j < N:
            result = mod(a*j, N)
        else:
            result = j
        for i in range(0,2**num_digits):
            if i == result:
                U[i,j] = 1
    return Qobj(U,dims = [[2 for i in range(num_digits)],[2 for i in range(num_digits)]])

def FidelityForeighthPhaseShift(rho,n_count):
    """
    :param rho: state of register after some QPE algorithm's run
    :param n_count: number of counting qubits
    :return: fidelity of register state to a run without errors
    """
    state = rho.ptrace([i for i in range(n_count)])
    #create right state for 1QPGO of 1/8 phase and n_count qubits and assuming |u> is the ground state
    right_state = tensor([basis(2,1)*basis(2,1).dag() if i==2 else basis(2,0)*basis(2,0).dag() for i in range(n_count)])
    return fidelity(state,right_state)

def sqrtM(U):
    """
    :param U: unitary 2x2 matrix
    :return: unitary matrix V such that VV=U
    """
    U = Qobj(U)
    if not (U.isunitary):
        return None
    u = np.array(U)
    return Qobj(scipy.linalg.sqrtm(u))

def ctrlParams(U):
    """
    :param U: single qubit gate (2x2 matrix)
    :return: real parameters alpha, beta, gamma, delta for decomposition of controlled-U by whitepaper
    """
    U=np.array(U)
    u=U[0,0]*U[1,1]-U[1,0]*U[0,1]
    a = U[0,0]
    b = U[0,1]
    c = U[1,0]
    alpha = -np.real((1/(2*1j)*np.log(u)))
    gamma = np.real(2*np.arccos(np.abs(a)))
    delta = np.real(np.angle(b)-np.angle(a) - np.pi)
    beta = np.real(np.angle(c)-np.angle(a))

    return (alpha, beta, gamma, delta)

def printQC(commands):
    for i,TimeCut in enumerate(commands):
        print('\n')
        if i==0:
            print('the gates in the '+str(i+1)+'st timecut are:')
        elif i==1:
            print('the gates in the '+str(i+1)+'nd timecut are:')
        elif i==2:
            print('the gates in the '+str(i+1)+'rd timecut are:')
        else:
            print('the gates in the '+str(i+1)+'th timecut are:')
        for command in TimeCut:
            gates = {'H','X','Y','Z','S','T'}
            rotations = {'Rx','Ry','Rz'}
            if command[0] in gates:
                print('gate '+str(command[0])+' acts on qubit '+str(command[1]))
            elif command[0] == 'CNOT':
                print('gate '+str(command[0])+' acts on qubit '+str(command[1])+ ' with qubit '+str(command[2])+' as control')
            elif command[0] in rotations:
                print('qubit '+str(command[1])+' rotates around the '+str(command[0][1])+' axis with angle '+str(command[3])+' [Rad]')
            else:
                print('this single qubit gate acts on qubit '+str(command[1])+':')
                print(command[3].data)
