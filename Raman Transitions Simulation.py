import numpy as np
import matplotlib.pyplot as plt
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution)

class system():

    def __init__(self,w02,w12,Omega_Rp,Omega_Rc,wp,wc):

        #self.hbar = 6.626 * 10**(-34) / (2 * np.pi)
        self.hbar = 1
        self.w02 = w02 #this gives energy spacing for hbar = 1
        self.w12 = w12
        self.w01 = self.w02 - self.w12
        self.Omega_Rp = Omega_Rp
        self.Omega_Rc = Omega_Rc
        self.wp = wp #Probe Frequency
        self.wc = wc #Coupling Frequency

        self.delta1 = self.wp - self.w02 #single photon detuning
        self.delta2 = self.wp - self.w01 - self.wc #2 photon detuning

        self.raise02 = fock(3, 2) * fock(3, 0).dag() #|2><0|
        self.raise12 = fock(3, 2) * fock(3, 1).dag() #|2><1|

    def MakeHamiltonian(self,doRWA,doAdiabaticElim):

        #We work in the interaction picture in a frame rotating with HD = hbar(wp - wc) |1><1| + hbar(wp) |2><2|
        
        if not doAdiabaticElim:

            if doRWA:

                #probe (0 -> 2)
                H_int_p = 0.5 * self.hbar * self.Omega_Rp * (self.raise02 + self.raise02.dag())
                #coupling (2 -> 1)
                H_int_c = 0.5 * self.hbar * self.Omega_Rc * (self.raise12 + self.raise12.dag())
                H_int = H_int_p + H_int_c
                H = -self.hbar * self.delta2 * fock_dm(3,1) - self.hbar * self.delta1 * fock_dm(3,2) + H_int

            else:

                #probe (0 -> 2)
                H_int_p1 = self.hbar * self.Omega_Rp * self.raise02
                def H_int_p1_coeff(t,args):
                    return 0.5 * (1 + np.exp(2j * self.wp * t))
                H_int_p2 = self.hbar * self.Omega_Rp * self.raise02.dag()
                def H_int_p2_coeff(t,args):
                    return 0.5 * (1 + np.exp(-2j * self.wp * t))
                
                #coupling (2 -> 1)
                H_int_c1 = self.hbar * self.Omega_Rc * self.raise12
                def H_int_c1_coeff(t,args):
                    return 0.5 * (1 + np.exp(2j * self.wc * t))
                H_int_c2 = self.hbar * self.Omega_Rc * self.raise12.dag()
                def H_int_c2_coeff(t,args):
                    return 0.5 * (1 + np.exp(-2j * self.wc * t))
                
                H = [-self.hbar * self.delta2 * fock_dm(3,1),- self.hbar * self.delta1 * fock_dm(3,2)
                     ,[H_int_p1,H_int_p1_coeff],[H_int_p2,H_int_p2_coeff]
                     ,[H_int_c1,H_int_c1_coeff],[H_int_c2,H_int_c2_coeff]]
                
        return H

    def TransformFrame(self,rhoOutTilda,times):
        rhoOut = []
        for i in range(len(times)):
            t = times[i]
            U = fock_dm(3,0) + fock_dm(3,2) * np.exp(1j * self.wp * t) + fock_dm(3,1) * np.exp(1j * (self.wp - self.wc) * t)
            rhoOut.append(U.dag() * rhoOutTilda[i] * U)
        return rhoOut

    def CreateCollapseOperators(self,gamma20,gamma21):
        L20_TI = np.sqrt(gamma20) * self.raise02.dag() #spontaneous emission from |2> to |0>
        def L20_coeff(t,args):
            return np.exp(-1j * self.wp * t)
        L21_TI = np.sqrt(gamma21) * self.raise12.dag() #spontaneous emission from |2> to |1>
        def L21_coeff(t,args):
            return np.exp(-1j * self.wc * t)

        L20 = [[L20_TI,L20_coeff]]
        L21 = [[L21_TI,L21_coeff]]

        return [L20,L21]
        
    
def PlotRabiOscillations(system,times,rhoIn,doRWA,doAdiabaticElim,AllowSpontEmission):

    H = system.MakeHamiltonian(doRWA,doAdiabaticElim)

    if(AllowSpontEmission):
        cOps = system.CreateCollapseOperators(gamma20,gamma21)
        result = mesolve(H,rhoIn,times,cOps,[])

    else:
        result = mesolve(H,rhoIn,times,[],[])
        
    #Need to transform out of the frame with exp(-iH_Dt/hbar) rhoOut exp(iH_Dt/hbar)
    rhoOut = system.TransformFrame(result.states,times)

    plt.plot(times,expect(fock_dm(3,1) - fock_dm(3,0),rhoOut), label='Expectation value of σz')
    plt.plot(times,expect(fock_dm(3,2),rhoOut),label='Expectation population of intermediate state')
    print('Producing plot of Rabi oscillations for Raman transitions with detunings Δ = ' + str(round(system.delta1, 3)) + '/s, δ = ' + str(round(system.delta2, 3))+ '/s.')
    plt.title('Rabi oscillations via Raman transitions')
    plt.ylabel('Expectation values of state populations')
    plt.xlabel('Time (s)')
    plt.legend(loc="lower right")
    plt.show()

def CalculateACSSDetuning(Omega_Rp,Omega_Rc,delta1,delta2):
    ACSS_detuning = (Omega_Rp**2 - Omega_Rc**2)/(4 * delta1)
    return (delta2 - ACSS_detuning)

#--------------------------------------------- Raman Simulation Parameters ------------------------------------------

doRWA = True
doAdiabaticElim = False
AllowSpontEmission = False
CorrectACSS = True #Correct against AC Stark Shift

#We require for 2 photon Raman transitions that:
#w02,w12 (optical) >> delta1 >> w01 (hyperfine splitting) >> Linewidth of optical transitions > delta2
#For off-resonance to |2>, also require delta1 >> Omega_Rp, Omega_Rc, gamma20, gamma21, delta2

Omega_Rp = 0.01*np.pi
Omega_Rc = 0.02*np.pi
w02 = 2.99 * np.pi
w12 = 3 * np.pi

delta1 = 0.1 * np.pi #Single photon detuning
delta2 = 0.0001 * np.pi #2 photon detuning

if(CorrectACSS):
    delta2 = CalculateACSSDetuning(Omega_Rp,Omega_Rc,delta1,delta2)

wp = delta1 + w02 #Probe Laser Frequency
wc = -delta2 + wp - (w02 - w12) #Coupling Laser Frequency

gamma20 = 0.005
gamma21 = 0.005

#Do simulation

sys = system(w02,w12,Omega_Rp,Omega_Rc,wp,wc)

times = np.linspace(0,10000,1000)
rhoIn = fock_dm(3,0)

PlotRabiOscillations(sys,times,rhoIn,doRWA,doAdiabaticElim,AllowSpontEmission)




    

