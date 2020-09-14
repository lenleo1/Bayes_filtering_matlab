%% UKF Application Example 
% Reference: 
% [1] Julier, SJ. and Uhlmann, J.K., Unscented Filtering and Nonlinear Estimation, Proceedings of the IEEE, Vol. 92, No. 3,pp.401-422, 2004. 
% [2] Yi Cao (2020). Learning the Unscented Kalman Filter, MATLAB Central File Exchange. Retrieved September 14, 2020.
% by lenleo
clear;clc
n=2;                                                                       % number of state
q=1;                                                                       % noise std of process 
r=1;                                                                       % noise std of measurement
Q=q^2*eye(n);                                                              % noise covariance of process
R=r^2;                                                                     % nlise covariance of measurement  
f=@(x)[2*sin(x(1))+x(2);2*cos(x(1))-x(2)];                                 % nonlinear state equations
h=@(x)x(1);

s=[0.2;0.2];                                                               % initial state
x=s+q*randn(n,1);                                                          % initial state with noise
P = eye(n);                                                                % initial state covraiance
N=50;                                                                      % total dynamic steps
xV = zeros(n,N);                                                           %allocate memory for estimate
sV = zeros(n,N);                                                           %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                                                      % measurments
  sV(:,k)= s;                                                              % save actual state
  zV(k)  = z;                                                              % save measurment
  [x, P] = ukf(f,x,P,h,z,Q,R);                                             % ukf 
  xV(:,k) = x;                                                             % save estimate
  s = f(s) + q*randn(n,1);                                                 % update process 
end
for k=1:n                                                                  % plot results
  subplot(n,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
  legend('actual state','estimate state');
end
%% UKF function
function [x,P]=ukf(fstate,x,P,hmeas,z,Q,R)
% UKF   Unscented Kalman Filter for nonlinear dynamic systems
% [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           P: "a priori" estimated state covariance
%           h: fanction handle for h(x)
%           z: current measurement
%           Q: process noise covariance 
%           R: measurement noise covariance
% Output:   x: "a posteriori" state estimate
%           P: "a posteriori" state covariance

L=numel(x);                                                                %numer of states
m=numel(z);                                                                %numer of measurements
% set tunable paras
alpha=1e-3;                                                                %default, tunable
ki=0;                                                                      %default, tunable
beta=2;                                                                    %default, tunable
lambda=alpha^2*(L+ki)-L;                                                   %scaling factor
c=L+lambda;                                                                %scaling factor
% choose sigma points xsigma
Wm=[lambda/c 0.5/c+zeros(1,2*L)];                                          %weights for means
Wc=Wm;
Wc(1)=Wc(1)+(1-alpha^2+beta);                                              %weights for covariance
c=sqrt(c);
xsigma=sigmas(x,P,c);                                                      %sigma points around x
[xhat_minus,xsigma_minus,P_minus,dx_minus]=ut(fstate,xsigma,Wm,Wc,L,Q);    %unscented transformation of process
[zhat,~,Py,dz]=ut(hmeas,xsigma_minus,Wm,Wc,m,R);                           %unscented transformation of measurments
Pxy=dx_minus*diag(Wc)*dz';                                                 %transformed cross-covariance
K=Pxy*inv(Py);                                                             %kalman gain
x=xhat_minus+K*(z-zhat);                                                   %state update
P=P_minus-K*Pxy';                                                          %covariance update

function [xhat_minus,xsigma_minus,P_minus,dx_minus]=ut(f,xsigma,Wm,Wc,n,R)
%Unscented Transformation
%Input:
%        f: nonlinear map
%       xsigma: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        R: additive covariance
%Output:
%        xhat_minus: transformed mean
%        xsigma_minus: transformed smapling points
%        P_minus: transformed covariance
%        dx_minus: transformed deviations
L=size(xsigma,2);
xhat_minus=zeros(n,1);
xsigma_minus=zeros(n,L);
for k=1:L                   
    xsigma_minus(:,k)=f(xsigma(:,k));       
    xhat_minus=xhat_minus+Wm(k)*xsigma_minus(:,k);       
end
dx_minus=xsigma_minus-xhat_minus(:,ones(1,L));
P_minus=dx_minus*diag(Wc)*dx_minus'+R;
end

function xsigma=sigmas(x,P,c)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points

A = c*chol(P)';
Y = x(:,ones(1,numel(x)));
xsigma = [x Y+A Y-A]; 
end
end