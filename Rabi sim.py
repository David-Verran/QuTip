import numpy as np
import matplotlib.pyplot as plt
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution)

class system():

    def __init__(self,w0,Omega_R,wLaser):

        #self.hbar = 6.626 * 10**(-34) / (2 * np.pi)
        self.hbar = 1
        self.w0 = w0 #this gives energy spacing for hbar = 1
        self.Omega_R = Omega_R
        self.wLaser = wLaser

    def MakeHamiltonian(self,RWA):

        #We assume a two level atom with energy spacing hbar w0
        H0 = -0.5 * self.hbar * self.w0 * sigmaz()

        #Electric dipole perturbation
        V_TI = self.hbar * self.Omega_R * sigmax()
        def V_TD(t,args):
            return np.cos(self.wLaser * t)

        #Apply RWA

        sigma_p = create(2)
        sigma_m = destroy(2)
        
        V_TI_RWA1 = self.hbar * self.Omega_R * sigma_p
        def V_TD_RWA1(t,args):
            return 0.5 * np.exp(-1j * self.wLaser * t)

        V_TI_RWA2 = self.hbar * self.Omega_R * sigma_m
        def V_TD_RWA2(t,args):
            return 0.5 * np.exp(1j * self.wLaser * t)

        #Create Hamiltonian
        
        if(not RWA):
            H_tot = [H0,[V_TI,V_TD]]
        else:
            H_tot = [H0,[V_TI_RWA1,V_TD_RWA1],[V_TI_RWA2,V_TD_RWA2]]

        return H_tot
    
def PlotRabiOscillations(system,times,rhoIn,Omega_R,detuning,doRWA):

    H = system.MakeHamiltonian(doRWA)
    result = mesolve(H,rhoIn,times,[],[])

    plt.plot(times,expect(sigmaz(),result.states))
    plt.title('Rabi oscillations for Rabi Frequency ' + str(round(Omega_R, 3)) + '/s, detuning ' + str(round(detuning, 3))+ '/s')
    plt.ylabel('Expectation value of sigma_z')
    plt.xlabel('Time')
    plt.show()
    
#sys = system(10**17,0.1,10**17)

Omega_R = 0.1*np.pi
w0 = 3 * np.pi
detuning = 0.0 * np.pi
Omega_Eff = np.sqrt(Omega_R**2 + detuning**2)
Amplitude = (Omega_R/Omega_Eff)**2 * 1 + (1-(Omega_R/Omega_Eff)**2) * -1
Period = 2* np.pi / Omega_Eff
sys = system(w0,Omega_R,w0 + detuning)
doRWA = False

times = np.linspace(0,Period * 4,1000)
rhoIn = fock_dm(2,1)

print('Expected amplitude is ' + str(Amplitude) + '.')
print('Expected period is ' + str(Period) + '.')

PlotRabiOscillations(sys,times,rhoIn,Omega_R,detuning,doRWA)




    

