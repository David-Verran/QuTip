import numpy as np
import matplotlib.pyplot as plt
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, sigmap, sigmam, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution, visualization, fidelity, ket2dm)

#Defining the Ion-trap Hamiltonian

class system():

    def __init__(self,w0,v0,Omega_R,eta,maxNumPhonons):

        #self.hbar = 6.626 * 10**(-34) / (2 * np.pi)
        self.hbar = 1
        self.w0 = w0 #this gives energy spacing for hbar = 1
        self.v0 = v0 #phonon energy spacing for COM mode.
        self.Omega_R = Omega_R #Rabi frequency (assumed same for both lasers)
        self.eta = eta #Lamb-Dicke parameter (also assumed same)
        self.maxNumPhonons = maxNumPhonons

        self.raise01 = fock(2,1) * fock(2,0).dag()
        
    def MakeHamiltonian(self,wL,phi,onlyAddressSecondIon=False):

        #Here we couple |0> and |1> directly, no Raman transitions via third state.
        #Assume 2 level picture, both computational states are electric dipole-connected.
        #Use interaction picture rotating with H0:

        '''H_atomic = self.hbar * self.w0 * fock(2,1) * fock(2,1).dag()
        H_mot = self.hbar * self.v0 * create(self.maxNumPhonons) * destroy(self.maxNumPhonons)

        H0 = [tensor(H_atomic,qeye(2),qeye(self.maxNumPhonons)) + \
              tensor(qeye(2),H_atomic,qeye(self.maxNumPhonons)) + \
              tensor(qeye(2),qeye(2),H_mot)]'''

        if len(wL) == 2 and applyRWA:

            def red_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))
            def red_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))
            def blue_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[1] + self.w0 + self.v0) * t + phi[1]))
            def blue_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[1] + self.w0 + self.v0) * t + phi[1]))
            def carrier_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0) * t + phi[0]))\
                       + np.exp(-1j * ((-wL[1] + self.w0) * t + phi[1]))
            def carrier_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0) * t + phi[0]))\
                       + np.exp(1j * ((-wL[1] + self.w0) * t + phi[1]))

        elif len(wL) == 2 and not applyRWA:

            def red_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))\
                        + np.exp(-1j * ((-wL[1] + self.w0 - self.v0) * t + phi[1]))
            def red_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))\
                       + np.exp(1j * ((-wL[1] + self.w0 - self.v0) * t + phi[1]))
            def blue_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0 + self.v0) * t + phi[0]))\
                       + np.exp(-1j * ((-wL[1] + self.w0 + self.v0) * t + phi[1]))
            def blue_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0 + self.v0) * t + phi[0]))\
                       + np.exp(1j * ((-wL[1] + self.w0 + self.v0) * t + phi[1]))
            def carrier_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0) * t + phi[0]))\
                       + np.exp(-1j * ((-wL[1] + self.w0) * t + phi[1]))
            def carrier_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0) * t + phi[0]))\
                       + np.exp(1j * ((-wL[1] + self.w0) * t + phi[1]))

        elif len(wL) == 1:

            def red_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))
            def red_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0 - self.v0) * t + phi[0]))
            def blue_sideband_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0 + self.v0) * t + phi[0]))
            def blue_sideband_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0 + self.v0) * t + phi[0]))
            def carrier_emit_coeff(t,args):
                return np.exp(-1j * ((-wL[0] + self.w0) * t + phi[0]))
            def carrier_absorb_coeff(t,args):
                return np.exp(1j * ((-wL[0] + self.w0) * t + phi[0]))

        else:
            print('Error, invalid number of lasers')
        
        #Red Sideband transition
        #Atom 1

        if not onlyAddressSecondIon:
            H_red1e_tindep = 0.5 * -1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(self.raise01.dag(),qeye(2),create(maxNumPhonons)) #emit
            H_red1a_tindep = 0.5 * 1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(self.raise01,qeye(2),destroy(maxNumPhonons)) #absorb

        #Atom 2
        H_red2e_tindep = 0.5 * -1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(qeye(2),self.raise01.dag(),create(maxNumPhonons))
        H_red2a_tindep = 0.5 * 1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(qeye(2),self.raise01,destroy(maxNumPhonons))

        #Combine:
        if not onlyAddressSecondIon:
            H_red = [[H_red1e_tindep + H_red2e_tindep,red_sideband_emit_coeff],
                    [H_red1a_tindep + H_red2a_tindep,red_sideband_absorb_coeff]]
        else:
            H_red = [[H_red2e_tindep,red_sideband_emit_coeff],[H_red2a_tindep,red_sideband_absorb_coeff]]

        #Blue Sideband transition
        #Atom 1

        if not onlyAddressSecondIon:
            H_blue1e_tindep = 0.5 * -1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(self.raise01.dag(),qeye(2),destroy(maxNumPhonons))
            H_blue1a_tindep = 0.5 * 1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(self.raise01,qeye(2),create(maxNumPhonons))

        #Atom 2
        H_blue2e_tindep = 0.5 * -1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(qeye(2),self.raise01.dag(),destroy(maxNumPhonons))
        H_blue2a_tindep = 0.5 * 1j * self.hbar * self.Omega_R * self.eta * \
                             tensor(qeye(2),self.raise01,create(maxNumPhonons))

        #Combine:
        if not onlyAddressSecondIon:
            H_blue = [[H_blue1e_tindep + H_blue2e_tindep,blue_sideband_emit_coeff],
                     [H_blue1a_tindep + H_blue2a_tindep,blue_sideband_absorb_coeff]]
        else:
            H_blue = [[H_blue2e_tindep,blue_sideband_emit_coeff],[H_blue2a_tindep,blue_sideband_absorb_coeff]]

        #Carrier transition
        if not onlyAddressSecondIon:
            H_carr_a_tindep = 0.5 * self.hbar * self.Omega_R * \
                            (tensor(self.raise01,qeye(2),qeye(maxNumPhonons)) \
                            + tensor(qeye(2),self.raise01,qeye(maxNumPhonons)))
            H_carr_e_tindep = 0.5 * self.hbar * self.Omega_R * \
                            (tensor(self.raise01.dag(),qeye(2),qeye(maxNumPhonons)) \
                            + tensor(qeye(2),self.raise01.dag(),qeye(maxNumPhonons)))

        else:
            H_carr_a_tindep = 0.5 * self.hbar * self.Omega_R * \
                            (tensor(qeye(2),self.raise01,qeye(maxNumPhonons)))
            H_carr_e_tindep = 0.5 * self.hbar * self.Omega_R * \
                            (tensor(qeye(2),self.raise01.dag(),qeye(maxNumPhonons)))

        H_carr = [[H_carr_a_tindep,carrier_absorb_coeff],[H_carr_e_tindep,carrier_emit_coeff]]
        #Construct total Hamiltonian:
        if includeCarrier:
            H_int = H_red + H_blue + H_carr
        else:
            H_int = H_red + H_blue
        return H_int

    def GetcOps(self,gammas):
        L_SE1 = tensor(np.sqrt(gammas[0]) * self.raise01.dag(), qeye(2),qeye(maxNumPhonons))
        L_SE2 = tensor(qeye(2), np.sqrt(gammas[0]) * self.raise01.dag(),qeye(maxNumPhonons))
        def L_SE_coeff(t,args):
            return np.exp(-1j * self.w0 * t)

        L_PD1 = tensor(np.sqrt(gammas[1]) * sigmaz(), qeye(2),qeye(maxNumPhonons))
        L_PD2 = tensor(qeye(2), np.sqrt(gammas[1]) * sigmaz(),qeye(maxNumPhonons))

        L_heating = tensor(qeye(2),qeye(2),np.sqrt(gammas[2])* create(maxNumPhonons))

        L = [[L_SE1,L_SE_coeff],[L_SE2,L_SE_coeff],L_PD1,L_PD2,L_heating]
        return L

def doHadamard(system,rho,tRot,cOps):

    #H = Rx(pi/2) Ry(pi/2) Rx(pi/2) Ry(pi/2) Rx (-pi/2)
    Rx2 = system.MakeHamiltonian([w0],[0],True)
    Ry2 = system.MakeHamiltonian([w0],[np.pi/2],True)
    
    rho = mesolve(Rx2,rho,np.linspace(0,3*tRot,100),cOps,[]).states[-1]
    rho = mesolve(Ry2,rho,np.linspace(0,tRot,100),cOps,[]).states[-1]
    rho = mesolve(Rx2,rho,np.linspace(0,tRot,100),cOps,[]).states[-1]
    rho = mesolve(Ry2,rho,np.linspace(0,tRot,100),cOps,[]).states[-1]
    rho = mesolve(Rx2,rho,np.linspace(0,tRot,100),cOps,[]).states[-1]
    return rho

def SimulateGate(system,gateFrequencies,gatePhases,gammas,times,rhoIn,inputType,numSamplesPerGate,correctOutput):

    if makeCNOT: #Implement a controlled phase gate from the MS gate using single qubit rotations on either side.

        #For all single qubit gates, we implement a rotation by pi/2.
        #This is achieved by undergoing carrier Rabi oscillations with t = pi/(2 * Omega_R)
        tRot = np.pi/(2 * Omega_R)
        Rx = system.MakeHamiltonian([w0],[0])
        Ry = system.MakeHamiltonian([w0],[np.pi/2])

        '''
        To implement the MS gate, first we diagonalise the MS unitary by transforming basis
        by a pi/2 rotation about the y-axis. We then apply a rotation by pi/2 about the
        z-axis to create the controlled-phase gate. Lastly, we apply Hadamard to either
        side on just the second ion to construct CNOT.
        '''

        cOps = system.GetcOps(gammas)
        
        rhoOut = doHadamard(system,rhoIn,tRot,cOps) #Hadamard
        
        rhoOut = mesolve(Ry,rhoOut,np.linspace(0,tRot,100),cOps,[]).states[-1] #Transform to diagonal basis

        H_MS = system.MakeHamiltonian(gateFrequencies,gatePhases) #Apply MS gate
        rhoOut = mesolve(H_MS,rhoOut,times,cOps,[]).states[numSamplesPerGate]
        
        rhoOut = mesolve(Ry,rhoOut,np.linspace(0,3*tRot,100),cOps,[]).states[-1] #Transform back to computational basis
        
        rhoOut = mesolve(Rx,rhoOut,np.linspace(0,3*tRot,100),cOps,[]).states[-1] #Apply Rz(-pi/2) to make C-Z
        rhoOut = mesolve(Ry,rhoOut,np.linspace(0,3*tRot,100),cOps,[]).states[-1]
        rhoOut = mesolve(Rx,rhoOut,np.linspace(0,tRot,100),cOps,[]).states[-1]
        
        rhoOut = doHadamard(system,rhoOut,tRot) #Hadamard

        #Trace out phonon space:
        rhoOut = rhoOut.ptrace([0,1])
        return(rhoOut)
        
    else:

        H = system.MakeHamiltonian(gateFrequencies,gatePhases)
        cOps = system.GetcOps(gammas)
        #Note this output is still in the rotating picture, no phase accrual due to time evolution under H0.
        result = mesolve(H,rhoIn,times,cOps,[])
        rhoOut = result.states[numSamplesPerGate].ptrace([0,1]) #Trace out the phonon space
        print('Fidelity for input ' + inputType + ' is ' + str(min(np.round(fidelity(rhoOut,correctOutput),5),1)))
        return result.states

def CalculateGateTime(Omega_R, eta, detuning):

    #The gate time will be a quarter of the Rabi oscillation time.
    #From adiabatic elimination, we find Omega_eff = (Omega_R**2 * eta**2)/(2 * detuning)
    
    Omega_eff = Omega_R**2 * eta**2 /(2 * detuning)    
    gateTime = np.pi/ (4 * Omega_eff)

    return gateTime

def CalculateDetuning(Omega_R,eta,m):

    #Fix detuning:
    #We choose delta based on 2 requirements:
    #Firstly, we require that the phase matching condition is met.
    #So delta * t_g = 2 * pi * m, for m an integer.
    #This means that for larger m, small errors in delta become amplified.
    #However, large detuning is favourable to keep Omega_R/Delta as small as possible.

    if(m%1 == 0):
        print('\nLoop closure condition is met.')
    else:
        print('\nLoop closure condition is NOT met.')
    
    delta = 2 * Omega_R * eta * np.sqrt(m) #detuning
    return delta
    
def RunSimulations():

    if plotAllInputs:
        fig, axs = plt.subplots(2, 2)
    
        if not makeCNOT:
            for i in range(2):
                for j in range(2):

                    #Define sigma_z for the ee,gg or eg,ge subspace:
                    sz = tensor(((tensor(fock(2,1-i),fock(2,1-j)))*(tensor(fock(2,1-i),fock(2,1-j)).dag())),qeye(maxNumPhonons)) \
                         - tensor(((tensor(fock(2,i),fock(2,j)))*(tensor(fock(2,i),fock(2,j)).dag())),qeye(maxNumPhonons))
                    rho = SimulateGate(sys,gateFrequencies,gatePhases,gammas,times,inputdms[i][j],inputLabels[i][j],numSamplesPerGate,correctOutputs[i][j])
                    #axs[i, j].plot(times,expect(tensor((fock(2,1)*fock(2,1).dag() - fock(2,0)*fock(2,0).dag()),qeye(2),qeye(maxNumPhonons)),rho), label="⟨σ_z⟩ for Atom 1")
                    #axs[i, j].plot(times,expect(tensor(qeye(2),(fock(2,1)*fock(2,1).dag() - fock(2,0)*fock(2,0).dag()),qeye(maxNumPhonons)),rho), label="⟨σ_z⟩ for Atom 2")
                    #axs[i, j].plot(times,AdiabaticEliminationPrediction(times), label="⟨σ_z⟩ prediction from adiabatic elimination")
                    axs[i, j].plot(times * 1000,expect(sz,rho), label="⟨σ_z⟩ for 2-qubit computational subspace")
                    axs[i, j].plot(times[numSamplesPerGate] * 1000,expect(sz,rho[numSamplesPerGate]), marker='x',color = 'red')
                    axs[i, j].plot(times * 1000,expect(tensor(qeye(2),qeye(2), create(maxNumPhonons) * destroy(maxNumPhonons)),rho), color='orange', label="Number of Phonons")
                    axs[i, j].plot(times * 1000, [1]*len(times), linestyle='dashed', markersize=1,color = 'green')
                    axs[i, j].text(times[numSamplesPerGate] * 1000,expect(sz,rho[numSamplesPerGate]), 'Gate Output', fontsize=8, horizontalalignment='left',verticalalignment='top')
                    axs[i, j].set_title('Rabi Oscillations for input = |' + inputLabels[i][j] + '⟩')
                    axs[i, j].legend(loc="lower right")
    
                #print(rho[len(times)-1].ptrace([0,1]))

            fig.suptitle('MS gate simulation: Ωᵣ = ' + str(np.round(sys.Omega_R,4)) + ', ω₀ = ' + str(np.round(sys.w0,4)) + ', ν₀ = ' + str(np.round(sys.v0,4)) + ', η = ' + str(np.round(sys.eta,4)) + ' and δ = '+ str(np.round(wL2 - sys.w0 - sys.v0,4)))
            for ax in axs.flat:
                ax.set(xlabel='Time (ms)', ylabel='Expectation values for populations')
                ax.label_outer()
            plt.show()

        else:
            for i in range(2):
                for j in range(2):
                    rho = SimulateGate(sys,gateFrequencies,gatePhases,gammas,times,inputdms[i][j],inputLabels[i][j],numSamplesPerGate,correctOutputs[i][j])
                    visualization.hinton(rho,ax=axs[i][j])
                    axs[i, j].set_title('C-NOT gate output for input |' + inputLabels[i][j] + '⟩')
            fig.suptitle('MS gate simulation: Ωᵣ = ' + str(np.round(sys.Omega_R,4)) + ', ω₀ = ' + str(np.round(sys.w0,4)) + ', ν₀ = ' + str(np.round(sys.v0,4)) + ', η = ' + str(np.round(sys.eta,4)) + ' and δ = '+ str(np.round(wL2 - sys.w0 - sys.v0,4)))
            plt.show()
    else:
        
        #we just plot the gg <-> ee transition
    
        if not makeCNOT:

            #Define sigma_z for the ee,gg or eg,ge subspace:
            sz = tensor(((tensor(fock(2,1),fock(2,1)))*(tensor(fock(2,1),fock(2,1)).dag())),qeye(maxNumPhonons)) \
                 - tensor(((tensor(fock(2,0),fock(2,0)))*(tensor(fock(2,0),fock(2,0)).dag())),qeye(maxNumPhonons))
            rho = SimulateGate(sys,gateFrequencies,gatePhases,gammas,times,inputdms[0][0],inputLabels[0][0],numSamplesPerGate,correctOutputs[0][0])
            plt.plot(times * 1000,expect(sz,rho), label="Expectation value of |ee⟩⟨ee| - |gg⟩⟨gg|")
            plt.plot(times * 1000,expect(tensor(qeye(2),qeye(2), create(maxNumPhonons) * destroy(maxNumPhonons)),rho), color='orange', label="Expected number of phonons")
            plt.plot(times * 1000, [1]*len(times), linestyle='dashed', markersize=1,color = 'green')
            if showGateOutput:
                plt.plot(times[numSamplesPerGate] * 1000,expect(tensor(qeye(2),qeye(2), create(maxNumPhonons) * destroy(maxNumPhonons)),rho[numSamplesPerGate]), marker='x',color = 'red')
                plt.plot(times[numSamplesPerGate] * 1000,expect(sz,rho[numSamplesPerGate]), marker='x',color = 'red')
                plt.text(times[numSamplesPerGate] * 1000,expect(sz,rho[numSamplesPerGate]), 'Gate Output', fontsize=15, horizontalalignment='left',verticalalignment='top')
            plt.title('Rabi oscillations for |gg⟩ ↔ |ee⟩ for Ωᵣ = ' + str(np.round(sys.Omega_R/(2 * np.pi * (10**6)),4)) + 'MHz, ν₀ = ' + str(np.round(sys.v0/(2 * np.pi * (10**6)),4)) + 'MHz, η = ' + str(np.round(sys.eta,4)) + ' and δ = '+ str(np.round((wL2 - sys.w0 - sys.v0)/(2 * np.pi * (10**3)),4)) + 'kHz', fontsize=20)
            plt.legend(loc="lower right", fontsize=15)
            plt.xlabel('Time (ms)', fontsize=15)
            plt.ylabel('Expectation values for populations', fontsize=15)
            plt.show()

        else:
            print('Error: Cannot make CNOT with just one input.')

#----------------------------------------Simulation Parameters---------------------------------

makeCNOT = False
applyRWA = False
includeCarrier = True

showGateOutput = True
plotAllInputs = False
#Note if makeCNOT, includeCarrier must be True.

if makeCNOT:
    print('Implementing C-NOT with MS gate')

Omega_R = 0.5 * 10**6
w0 = 2585672965.92 * 10**6
v0 = 2 * np.pi * 10**6
eta = 0.06

#Noise
#spontaneous emission, phase damping, heating:
gammas = [0,0,0.5* 10**3]

avgNumPhonons = 0.2 #Assume cooled very close to zero.
maxNumPhonons = int(np.ceil(avgNumPhonons)) + 14 #Allow for many phonons to avoid truncation error.

m = 3 #Detuning parameter.
print('m = ' + str(m))
delta = CalculateDetuning(Omega_R,eta,m)
print('Generating MS gate simulation with Rabi frequency ' + str(np.round(Omega_R,4)) + ', qubit energy spacing ' + str(np.round(w0,4)) + ', COM mode phonon energy ' + str(np.round(v0,4)) + ', Raman detuning ' + str(np.round(delta,4)) + ' and Lamb-Dicke factor ' + str(np.round(eta,4)) + '.')

#For adiabatic regime, need detuning to be large compared to sideband and carrier Rabi frequencies.
#We want these all to be large:
print('\nδ/(Ωᵣ * η * sqrt(n)/2) = ' + str(np.round(delta/(0.5 * Omega_R * eta * np.sqrt(avgNumPhonons + 1)),4)))
print('(ν₀ + δ) / Ω_eff = ' + str(np.round((v0 + delta) * 2 * delta/(Omega_R**2 * eta**2),4)))
print('(ν₀ + δ) / Ωᵣ = ' + str(np.round((v0 + delta)/Omega_R,4)))

#For the MS-gate, we tune one laser to the red sideband transition, and one to the blue sideband transition.
wL1 = w0 - v0 - delta #Laser 1 Frequency (red sb)
wL2 = w0 + v0 + delta #Laser 2 Frequency (blue sb)
gateFrequencies = [wL1,wL2]

phiL1 = np.pi/2
phiL2 = np.pi/2 #We need the phases to add to pi.
gatePhases = [phiL1,phiL2]

sys = system(w0,v0,Omega_R,eta,maxNumPhonons)

#Define input density matrices:
#(note the initial phonon state can be anything for MS, but it's better if it's lower to minimise truncation errors, as well as carrier effects.)
rhoIn_gg = tensor(fock_dm(2,0),fock_dm(2,0),thermal_dm(maxNumPhonons,avgNumPhonons))
rhoIn_ge = tensor(fock_dm(2,0),fock_dm(2,1),thermal_dm(maxNumPhonons,avgNumPhonons))
rhoIn_eg = tensor(fock_dm(2,1),fock_dm(2,0),thermal_dm(maxNumPhonons,avgNumPhonons))
rhoIn_ee = tensor(fock_dm(2,1),fock_dm(2,1),thermal_dm(maxNumPhonons,avgNumPhonons))
inputdms = [[rhoIn_gg,rhoIn_ge],[rhoIn_eg,rhoIn_ee]]
inputLabels = [['gg','ge'],['eg','ee']]

#Calculate gate time
gateTime = CalculateGateTime(Omega_R, eta, delta)
print('\nGate time is ' + str(np.round(gateTime * 1000,2)) + ' ms\n')

#Need to ensure the times to simulate includes the gate time:
numSamplesPerGate = 1000
numGatesToSimulate = 8

tspacing = gateTime/numSamplesPerGate
maxTime = numGatesToSimulate * gateTime
numsamples = numGatesToSimulate * numSamplesPerGate + 1
times = np.linspace(0,maxTime,numsamples)

#Define expected outputs:

correctOutputs = [[ket2dm(tensor(fock(2,0),fock(2,0)) - 1j * tensor(fock(2,1),fock(2,1))).unit(),
                   ket2dm(tensor(fock(2,0),fock(2,1)) - 1j * tensor(fock(2,1),fock(2,0))).unit()],
                  [ket2dm(tensor(fock(2,1),fock(2,0)) - 1j * tensor(fock(2,0),fock(2,1))).unit(),
                   ket2dm(tensor(fock(2,1),fock(2,1)) - 1j * tensor(fock(2,0),fock(2,0))).unit()]]

RunSimulations()
