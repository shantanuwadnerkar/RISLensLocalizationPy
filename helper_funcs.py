#!/usr/bin/env python3

import math
import numpy as np
import numpy.matlib


# function showContours(signal,UE,RIS)
# % PEB computation in space
# % (c) 2020, Henk Wymeersch, henkw@chalmers.se

#     x_grid=linspace(-2,2,50);
#     z_grid=linspace(0.001,4,50);
#     Omega_b=diag(conj(RIS.b./(abs(RIS.b))));
#     bnew=RIS.b.'*Omega_b;    
#     PEB=zeros(length(x_grid),length(z_grid));
#     SNRr=zeros(length(x_grid),length(z_grid));
#     SNRd=zeros(length(x_grid),length(z_grid));
#     SNRp=zeros(length(x_grid),length(z_grid));

#     UE1=UE;
#     UE1.Location=0.1*[1;1;1];       % assume beams are pointing here        
#     UE1.rho=norm(UE1.Location);
#     UE1.phi=atan2(UE1.Location(2),UE1.Location(1));  % between 0 and 2pi
#     UE1.theta=acos(UE1.Location(3)/norm(UE1.Location)); % between 0 and pi (for us pi/2, since Z>0)  
#     UE1.mean=UE1.Location;   
#     UE1.covariance=UE1.covariance;        
#     BeamsR=getBeams(UE1,RIS,signal,'random');
#     BeamsD=getBeams(UE1,RIS,signal,'direction');        
#     BeamsP=getBeams(UE1,RIS,signal,'position');        
#     UE.Location=zeros(3,1);
  
#     for k=1:length(x_grid)                  
#         UE.Location(1)=x_grid(k);
#         UE.Location(2)=x_grid(k);
#         for l=1:length(z_grid)        
#             UE.Location(3)=z_grid(l);
#             UE.rho=norm(UE.Location);
#             UE.phi=atan2(UE.Location(2),UE.Location(1));  % between 0 and 2pi
#             UE.theta=acos(UE.Location(3)/norm(UE.Location)); % between 0 and pi (for us pi/2, since Z>0)       
#             h=computeRISChannel(UE.Location,RIS,signal,'CM3');           
                        
#             % random beams
#             W=bnew.*BeamsR;                                 
#             SNRr(k,l)=10*log10(norm(sqrt(signal.P)*W*h)^2/(2*signal.sigma2*signal.T));                                   
            
#             % directional beams
#             W=bnew.*BeamsD;                           
#             SNRd(k,l)=10*log10(norm(sqrt(signal.P)*W*h)^2/(2*signal.sigma2*signal.T));            
            
#             % positional beams
#             W=bnew.*BeamsP;                          
#             SNRp(k,l)=10*log10(norm(sqrt(signal.P)*W*h)^2/(2*signal.sigma2*signal.T));            

#         end
#     end
  
#         figure(2);
#         subplot(1,3,1)
#         SNRth=0;    % SNR threshold
#         f1=contourf(x_grid,z_grid,max(SNRr',SNRth),'edgecolor','none');
#         xl=xlabel('$x$ [m]');
#         yl=ylabel('$z$ [m]');
#         tt=title('random, SNR [dB]');
#         set(xl,'Interpreter','latex','FontSize',12);
#         set(yl,'Interpreter','latex','FontSize',12);
#         set(tt,'Interpreter','latex','FontSize',12);
#         pbaspect([2 2 1])
#         c = colorbar;
        

#         subplot(1,3,2)
#         f2=contourf(x_grid,z_grid,max(SNRd',SNRth),'edgecolor','none');
#         xl=xlabel('$x$ [m]');
#         yl=ylabel('$z$ [m]');
#         tt=title('directional, SNR [dB]');
#         set(xl,'Interpreter','latex','FontSize',12);
#         set(yl,'Interpreter','latex','FontSize',12);
#         set(tt,'Interpreter','latex','FontSize',12);
#         pbaspect([2 2 1])
#         c = colorbar;
        


#         subplot(1,3,3)
#         f3=contourf(x_grid,z_grid,max(SNRp',SNRth),'edgecolor','none');
#         xl=xlabel('$x$ [m]');
#         yl=ylabel('$z$ [m]');
#         tt=title('positional, SNR [dB]');
#         set(xl,'Interpreter','latex','FontSize',12);
#         set(yl,'Interpreter','latex','FontSize',12);
#         set(tt,'Interpreter','latex','FontSize',12);
#         c = colorbar;
#         pbaspect([2 2 1])
#         set(gcf, 'Color', 'w');



# function showPEB(signal,UE,RIS,e_unit)
# % PEB in line
# % (c) 2020, Henk Wymeersch, henkw@chalmers.se
# signal.sims=2;
# PEBr=zeros(length(UE.distances),signal.sims);
# PEBd=zeros(length(UE.distances),signal.sims);
# PEBp=zeros(length(UE.distances),signal.sims);
# UE.covariance=(eye(3));
# for UEDi=1:length(UE.distances)        
#      for sim=1:signal.sims
#         dd=UE.distances(UEDi);
#         % UE parameters 
#         % --------------
#         UE.Location=dd*e_unit; 
#         UE.rho=norm(UE.Location);
#         UE.phi=atan2(UE.Location(2),UE.Location(1));  % between 0 and 2pi
#         UE.theta=acos(UE.Location(3)/norm(UE.Location)); % between 0 and pi (for us pi/2, since Z>0)
#         UE.mean=UE.Location;          
#         % Generate observation
#         % --------------------
#         h=computeRISChannel(UE.Location,RIS,signal,'CM2');
#         Omega_b=diag(conj(RIS.b./(abs(RIS.b))));
#         bnew=RIS.b.'*Omega_b;
        
#         Beams=getBeams(UE,RIS,signal,'random');
#         W=bnew.*Beams;
#         PEBr(UEDi,sim)=computePEB(UE,RIS,signal,W,'CM2');         
        
#          Beams=getBeams(UE,RIS,signal,'direction');
#         W=bnew.*Beams;            
#         PEBd(UEDi,sim)=computePEB(UE,RIS,signal,W,'CM2');         
        
#         Beams=getBeams(UE,RIS,signal,'position');
#         W=bnew.*Beams;                   
#         PEBp(UEDi,sim)=computePEB(UE,RIS,signal,W,'CM2'); 
        
        
#         % now also with concentrated a priori information
#         UE1=UE;
#         UE1.covariance=0.01*(eye(3));
        
#         Beams=getBeams(UE1,RIS,signal,'direction');
#         W=bnew.*Beams;            
#         PEBd2(UEDi,sim)=computePEB(UE1,RIS,signal,W,'CM2');         
        
#         Beams=getBeams(UE1,RIS,signal,'position');
#         W=bnew.*Beams;                   
#         PEBp2(UEDi,sim)=computePEB(UE1,RIS,signal,W,'CM2'); 

        
#      end
# end

# figure(1)
# semilogy(UE.distances,mean(PEBr'),'s-',UE.distances,mean(PEBd'),'+-',UE.distances,mean(PEBp'),'*-',UE.distances,mean(PEBd2'),'+--',UE.distances,mean(PEBp2'),'*--')
# xlabel('distance [m]')
# ylabel('PEB [m]')
# grid
# legend('randomized','directional, \sigma = 1 ','positional, \sigma = 1','directional, \sigma = 0.1','positional, \sigma = 0.1','FontSize',12)



def quantizePhases(phases_in, nbits):
    if nbits > 5:
        phases_out = phases_in
    else:
        delta = np.pi / nbits
        phases_out = np.floor(phases_in / delta + 0.5) * delta
    
    return phases_out


def computeLogLikelihood(a, y, W, Signal):
    gamma = math.sqrt(Signal.P) * W * a
    llf = - np.linalg.norm(y - (np.transpose(gamma) * y)) / pow(2, np.linalg.norm(pow(2, gamma)) * gamma) / Signal.sigma2
    return llf


# function PEB = computePEB(UE,RIS,signal,W,regime)
def computePEB(UE, RIS, Signal, W, regime):
    
    h, RIS_phases, locations = computeRISChannel(UE.Location, RIS, Signal, regime)
    #     a=exp(-1j*RISphases);       
    #     K=locations-UE.Location;
    #     d=vecnorm(K);
    #     K=K./d;
    #     er=UE.Location/norm(UE.Location);
    #     Da=1j*2*pi/signal.lambda*(diag(a)*K'+a*er');     
    #     rho=abs(h(1));
    #     J=zeros(5,5);   
    #     for t=1:signal.T                
    #         myGradient=sqrt(signal.P)*W(t,:)*[a 1j*rho*a rho*Da];
    #         J=J+real(myGradient'*myGradient);
    #     end        
    #     J=J/signal.sigma2;
    #     Jinv=inv(J);
    #     PEB=sqrt(trace(Jinv(3:5,3:5)));  
    pass


# function [gain, phase_rot, locations]=computeRISChannel(source,IRS,signal,regime)
def computeRISChannel(source, IRS, Signal, regime):
    M = IRS.M
    fc = Signal.fc
    c = Signal.c
    spacing = IRS.Delta
    RIS = IRS.Location
    Lambda = c / fc             # Wavelenght(m)
    A = pow(2, spacing)         # basic element area
    a = math.sqrt(A)            # basic element size

    phi = math.atan2(source[1], source[0])
    theta = math.acos(source[2] / np.linalg.norm(source))
    k = 2 * np.pi / Lambda * np.array([math.cos(phi)*math.sin(theta), math.sin(phi)*math.sin(theta), math.cos(theta)]).reshape((3, 1))
    gain = np.zeros((M, 1))
    phase_rot = np.zeros((M, 1))

    iix = np.linspace(0, math.sqrt(M) - 1)
    iiy = np.linspace(0, math.sqrt(M) - 1)
    iix = a * (iix - math.sqrt(M) / 2)
    iiy = a * (iiy - math.sqrt(M) / 2)

    locations = np.zeros((3, M))
    
    ###### verify below this
    coord = np.matlib.repmat(np.linspace(1, math.sqrt(M), dtype=int), int(math.sqrt(M)), 1)
    x_temp = iix[coord - 1]
    y_temp = iiy[np.transpose(coord) - 1]
    xyz_mat = np.array([np.transpose(x_temp), np.transpose(y_temp)])
    ###### verify above this


    # locations(1:2,:)=XYZmat;
    # d=vecnorm(locations-source);
    # d0=norm(source);

    correction = 1- pow(2, math.sin(theta)) * pow(2, math.sin(phi))

    if regime == "CM1":
    #         phase_rot=mod(-k'*locations,2*pi);
    #         phase_rot=phase_rot';
    #         gain=((sqrt(cos(theta)*correction)*a)/(sqrt(4*pi)*norm(source-RIS)))*exp(-1i*phase_rot);
        pass
    elif regime == "CM2":
    #         phase_rot=mod(2*pi*(d-d0)/lambda,2*pi);        
    #         phase_rot=phase_rot';
    #         gain=((sqrt(cos(theta)*correction)*a)/(sqrt(4*pi)*norm(source-RIS)))*exp(-1i*phase_rot);
        pass
    elif regime == "CM3":
    #     for m=1:M          
    #         el_m =locations(:,m);
    #         x_m=el_m(1);
    #         y_m=el_m(2);       
    #         setX=[a/2+x_m-source(1) a/2-x_m+source(1)];             
    #         setY=[a/2+y_m-source(2) a/2-y_m+source(2)];                                              
    #         d=abs(source(3));                            % according to the paper, the Z-coordinate is d
    #         TMP=zeros(length(setX),length(setY));
    #         dm=norm(source-el_m);                   % distance between BS and m-th element
    #         for ix=1:length(setX)
    #             for iy=1:length(setY)                        
    #                 a2=(setX(ix)*setY(iy))/d^2;                        
    #                 b21=3*((setY(iy)^2)/d^2)+3;                        
    #                 b22=sqrt(((setX(ix)^2+setY(iy)^2)/d^2)+1);                       
    #                 TMP(ix,iy)=(a2/(b21*b22))+((2/3)*atan2(a2,b22));                        
    #             end
    #         end                     
    #         powerval=((1/(4*pi))*sum(sum(TMP)));                % power of the m-th element                         
    #         phase_rot(m)=2*pi*(dm-d0)/lambda;
    #         gain(m)=sqrt(powerval)*exp(-1i*phase_rot(m));     % complex channel gain                               
        pass


    return (gain, phase_rot, locations)


# function UE=getAngleEstimateSimple(UE,RIS,signal,y,W)
def getAngleEstimateSimple(UE, RIS, Signal, y, W):
    #     % create a grid
    #     phi_grid=linspace(UE.phiRange(1),UE.phiRange(2),360);
    #     theta_grid=linspace(UE.thetaRange(1),UE.thetaRange(2),90);                        
        
    #     % first estimate theta
    #     [~,~,RISlocations]=computeRISChannel([1;1;1],RIS,signal,'flat');    
    #     r=vecnorm(RISlocations);
    #     psi=atan2(RISlocations(2,:),RISlocations(1,:));       
    #     N=5;        % number of terms in the expansion
    #     A=zeros(signal.T,2*N+1,length(theta_grid));    
    #     for k2=1:length(theta_grid)
    #         theta=theta_grid(k2);
    #         % make G(theta) matrix
    #         G=zeros(2*N+1,RIS.M);
    #         for n=-N:N
    #             % make 1 row of G
    #             g=1j^n*exp(-1j*n*psi).*besselj(n,-2*pi*r*sin(theta)/signal.lambda);
    #             G(n+N+1,:)=g;
    #         end 
    #         if (signal.T>N)        
    #             A(:,:,k2)=W*G.';  % TxN
    #             B=A(:,:,k2)'*A(:,:,k2); % NxN
    #             cost(k2)=norm(y-A(:,:,k2)*inv(B)*A(:,:,k2)'*y);
    #         else
    #             disp('error: too few observations')
    #             keyboard
    #         end
    #     end
    #     [mv,mi]=min(cost);
    #     UE.bestTheta=theta_grid(mi);
        
    #     % now estimate phi:      
    #     LogWeight=zeros(1,length(phi_grid));
    #     for k=1:length(phi_grid)
    #         p=[sin(UE.bestTheta)*cos(phi_grid(k)); sin(UE.bestTheta)*sin(phi_grid(k)); cos(UE.bestTheta)];                  
    #         [~,RISphases]=computeRISChannel(p,RIS,signal,'flat');        
    #         LogWeight(k)=computeLogLikelihood(exp(-1j*RISphases),y,W,signal);       
    #     end    
    #     [mv,mi]=max(LogWeight);
    #     UE.bestPhi=phi_grid(mi);
    pass


# function Beams=getBeams(UE,RIS,signal,beamType)
def getBeams(UE, RIS, Signal, beam_type):
    
    if beam_type == "random":
        phases = np.random.uniform(size=(5, 2)) * 2 * np.pi
        phases = quantizePhases(phases, RIS.bits)
        beams = np.exp(1j * phases)
    elif beam_type == "direction":
        # P = mvnrnd(UE.mean, UE.covariance, Signal.T)'
        # ii = P(3, :) < 0
        # P(3, ii) = P(3, ii) * -1
        pass
    elif beam_type == "position":
    #             P=mvnrnd(UE.mean,UE.covariance,signal.T)';
    #             ii=P(3,:)<0;
    #             P(3,ii)=P(3,ii)*(-1);     % only positive Z axis                           
    #             for t=1:signal.T            
    #                 [~,RISphases]=computeRISChannel(P(:,t),RIS,signal,'CM1');
    #                 RISphases=quantizePhases(RISphases,RIS.bits);
    #                 a=exp(-1j*RISphases);        
    #                 a=a';       % make conjugate
    #                 Beams(t,:)=a;
        pass
    beams = 0
    return beams


# function [Pest,LLF]=getPositionSimple(UE,RIS,signal,y,W,regime)
def getPositionSimple(UE, RIS, Signal, y, W, regime):
    delta = 0.01
    p_num = round((UE.rhoRange[1] - UE.rhoRange[0])) / delta
    p_num = min(2000, p_num)
    rho_grid = np.linspace(UE.rhoRange[0], UE.rhoRange[1], p_num)
    plog_weight = np.zeros((1, p_num))
    #     % generate possible locations
    #     Ploc=rho_grid.*[cos(UE.bestPhi).*sin(UE.bestTheta); sin(UE.bestPhi).*sin(UE.bestTheta); cos(UE.bestTheta)];            
    #     for k=1:Pnum            
    #         % compute likelihood
    #         [~,RISphases]=computeRISChannel(Ploc(:,k),RIS,signal,regime);        
    #         PLogweight(k)=computeLogLikelihood(exp(-1j*RISphases),y,W,signal);           
    #     end                
    #     [LLF,index]=max(PLogweight);
    #     Pest=Ploc(:,index);
    pass
