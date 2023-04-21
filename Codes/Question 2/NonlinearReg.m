%% Masoud Pourghavam
%% Student Number: 810601044
%% Course: Artificial Intelligence
%% University of Tehran


clc;
clear all;
close all;

data = xlsread('Performance-Degradation Data Nelson.xlsx');
y = data(:,1);
x1 = data(:,2);
x2 = data(:,3);
fun = @(b,x) b(1) - b(2)*x(:,1).*exp(-b(3)*x(:,2));
beta0 = [1 1 1]; % initial guess for beta
beta = lsqcurvefit(fun,beta0,[x1 x2],log(y));
b1 = beta(1);
b2 = beta(2);
b3 = beta(3);
y_fit = exp(fun(beta,[x1 x2])); % calculate fitted values
ss_res = sum((log(y) - fun(beta,[x1 x2])).^2); % calculate residual sum of squares
ss_tot = sum((log(y) - mean(log(y))).^2); % calculate total sum of squares
R2 = 1 - ss_res/ss_tot; % calculate R2-score
RMSE = sqrt(ss_res/length(y)); % calculate root mean square error
x2_fit = linspace(min(x2),max(x2),100);
x1_fit = linspace(min(x1),max(x1),100);
[X1,X2] = meshgrid(x1_fit,x2_fit);
Y_fit = exp(fun(beta,[X1(:) X2(:)]));
figure;
scatter3(x1,x2,y,'filled', "blue");
hold on;
mesh(X1,X2,  reshape(Y_fit,100,100));
colormap('autumn');
legend ('Data','Fitted');
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Nonlinear regression with Fitted Function');

disp(['b1: ', num2str(beta(1))]);
disp(['b2: ', num2str(beta(2))]);
disp(['b3: ', num2str(beta(3))]);
disp(['R2-score: ', num2str(R2)]);
disp(['Root Mean Square Error: ', num2str(RMSE)]);