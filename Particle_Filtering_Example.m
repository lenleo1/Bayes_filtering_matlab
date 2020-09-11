% Reference- Bilibili ID 287989852
% Nonlinear dynamics example
% x(i) = sin(x(i-1)) + 5*x(i-1)/(x(i-1)^2+1) + Q
% y(i) = x(i)^2 + R

% generate 100 samples
t = 0.01:0.01:1;
x = zeros(1,100);
y = zeros(1,100);

% set initial values
x(1) = 0.1;
y(1) = 0.01^2;
Q = 0.1;
R = 0.1;
% generate real data and mea data
for i = 2:100
    x(i) = 1*sin(x(i-1)) + 5*x(i-1)/(x(i-1)^2+1);
    y(i) = x(i)^2 + normrnd(0,1);
end
% PF start
% set particle sets
n = 150;
% xold = zeros(1,n);
xnew = zeros(1,n);
xplus = zeros(1,100); % xplus is for storing filtered value
% w = zeros(1,n);
% set x0 value and its weight
xold = zeros(1,n)+0.1;
w = zeros(1,n)+ 1/n;
% PF code
for i = 2:100
    % prediction, from x0 to x1
    for j=1:n
        xold(j) = 1*sin(xold(j))+5*xold(j)/(xold(j)^2+1)+normrnd(0,Q);
    end
    % update
    for j=1:n
        w(j) = exp(-((y(i)-xold(j)^2)^2/(2*R)));
    end
    % normlization
    w = w/sum(w);
    % re-sample
    c = zeros(1,n); % generate zone indexs
    c(1) = w(1);
    for j = 2:n
        c(j) = c(j-1) + w(j);
    end
    for j = 1:n
        a = unifrnd(0,1);% generate random numbers
        for k = 1:n
            if (a<c(k))
                xnew(j) = xold(k);
                break; % break is essential, Otherwise the resampled particle will be covered by the last particle
            end
        end
    end
    % end of re-sample
    % new value assigned to old value, iteraion goes on
    xold = xnew;
    % set all weights to 1/n
    w = zeros(1,n)+ 1/n;
    % assign the posterior probability expectation to xplus
    xplus(i) = sum(xnew)/n;
end
%%
figure(1)
plot(t,x,'r--',t,real(sqrt(y)),'k-.',t,xplus,'b-','LineWidth',1)
legend('real','mea','filtered')
