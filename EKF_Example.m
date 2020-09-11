%% generate real signal and measurements
clear;clc;
t = 0.01:0.01:1;
n = length(t);
x = zeros(1,n);
y = zeros(1,n);
x(1) = 0.1;
y(1) = 0.1;
for i= 2:n
    x(i) = sin(3*x(i-1))+ 1*x(i-1);
    y(i) = x(i)^2 + normrnd(0,1);
end
figure(1)
subplot(2,1,1)
plot(t,x,'r',t,sqrt(y),'b','LineWidth',2);
legend('real','mea');
%%
% set initial values
Pplus = 0.1;
Xplus(1) = 0.1;
Q = 0.01;
R = 1;
for i = 2:n
    % 1-prediction
    A = 3*cos(3*Xplus(i-1)) + 1;
    Xminus = sin(3*Xplus(i-1)) + Xplus(i-1);
    Pminus = A*Pplus*A' + Q;
    % 2-update
    C = 2*Xminus;
    K = C*Pminus*inv(C*Pminus*C' + R);
    Xplus(i) = Xminus + K*(y(i)-Xminus^2);
    Pplus = (eye(1)-K*C)*Pminus;
end
%%
figure(1)
subplot(2,1,2)
plot(t,x,'r',t,sqrt(y),'k',t,Xplus,'b','LineWidth',2)
legend('real','mea','EKF');

    
    