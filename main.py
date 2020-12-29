#!/usr/bin/env python3

# Source: https://github.com/henkwymeersch/RISLensLocalization
# Convert Matlab code to Python

import numpy as np

import helper_funcs


class Signal:
    fc = 28                                # carrier in GHz
    c = 0.3                                # speed of light [m/ns]
    T = 200                                # number of observations
    P = 1                                  # transmit power mW
    N0dBmHz = -174                         # dBm/Hz
    N0 = pow(10, 0.1 * N0dBmHz) * 1e9 # noise PSD  [mW/GHz] (290 Kelvin * Boltzmann constant in W/Hz)
    BW = 1e-3                              # Bandwidth GHz
    NFdB = 8                               # receiver noise figure [dB]
    NF = pow(10, 0.1 * NFdB)
    EsN0 = 2 * P / (NF * N0 * BW)
    sigma2 = NF * N0 * BW/2
    Lambda = c / fc                        # wavelength
    plot = 0                               # 1 = show plots
    sims = 5                               # number of Monte Carlo simulations


class RIS:
    M = 50 * 50                                                         # RIS elements 
    Delta = Signal.Lambda/2                                             # RIS element spacing
    Location = np.zeros((3, 1))                                         # location of RIS in XY plane
    Antenna = np.array([0, 0, -1 * Signal.Lambda]).reshape((3, 1))      # RIS antenna location
    Orientation = 0                                                     # RIS rotation around the Y axis 
    b = 0                                                               # response from RIS to antenna
    bits = 1000
    beam_type = "random"                                                # 'position', 'direction', 'random'


class UE:
    covariance = np.eye(3, dtype=int)               # UE a priori covariance
    distances = np.array([0.15, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15], dtype=float).reshape((1, 11))
    e = np.array([1, 1, 1])
    e_unit = e / np.linalg.norm(e)
    rhoRange = []
    phiRange = []
    thetaRange = []


RIS.b = helper_funcs.computeRISChannel(RIS.Antenna, RIS, Signal, "CM3") # response from RIS to antenna


beam_type = "random"
helper_funcs.getBeams(UE, RIS, Signal, beam_type)


# % these two function show the first and second figure from the paper
# showContours(signal,UE,RIS)
# showPEB(signal,UE,RIS,e_unit)


# % wipe of phase of b
# ab=helper_funcs.quantizePhases(angle(RIS.b),RIS.bits);        
# Omega_b=diag(exp(-1j*ab));        
# bnew=RIS.b.'*Omega_b;
# errorStat=zeros(length(UE.distances),signal.sims);
# RMSE=zeros(1,length(UE.distances));
# for UEDi=1:length(UE.distances)        
#     dd=UE.distances(UEDi);
#     for sim=1:signal.sims        
#         text=['simulation ' num2str(sim) ' for distance ' num2str(dd)];
#         disp(text)
#         % UE parameters 
#         % --------------
#         UE.Location=dd*e_unit; 
#         UE.rho=norm(UE.Location);
#         UE.phi=atan2(UE.Location(2),UE.Location(1));        % between 0 and 2pi
#         UE.theta=acos(UE.Location(3)/norm(UE.Location));    % between 0 and pi (for us pi/2, since Z>0)        
#         UE.mean=mvnrnd(UE.Location,UE.covariance)';           
#         UE.mean(3)=abs(UE.mean(3)); % Z coordinate is positive
        
#         % Generate observation
#         % --------------------
#         h=helper_funcs.computeRISChannel(UE.Location,RIS,signal,'CM3');                                
#         Beams=helper_funcs.getBeams(UE,RIS,signal,RIS.beamType);
#         W=bnew.*Beams;
#         noise=randn(signal.T,1)+1j*randn(signal.T,1);
#         s=sqrt(signal.P)*W*h;
#         y=s+noise*sqrt(signal.sigma2);  
        
#         % Estimate the position
#         % ----------------------
#         UE.rhoRange=[max(0,UE.rho-5) UE.rho+5];                
#         UE.phiRange=[0 2*pi];
#         UE.thetaRange=[0 pi/2];        
#         UE=helper_funcs.getAngleEstimateSimple(UE,RIS,signal,y,W);                               
#         [UE.estimate,LLF2]=helper_funcs.getPositionSimple(UE,RIS,signal,y,W,'CM2');        
#         errorStat(UEDi,sim)=norm(UE.estimate-UE.Location);               
#     end
#     yr=errorStat(UEDi,:).^2;    
#     RMSE(UEDi)=sqrt(mean(yr'));  
#     figure(5)
#     semilogy(UE.distances,RMSE)
#     ylabel('RMSE')
#     xlabel('distance to RIS')
#     grid
# end
