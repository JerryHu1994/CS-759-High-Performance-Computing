X = [5 6 7 8 9 10 11 12];
Y = [0.000535 0.000622 0.000504 0.000686 0.002834 0.003388 0.006063 0.012712];
logY = log10(Y);
p = polyfit(X, logY, 1);
regression_x = linspace(5,12);
regression_y = polyval(p,regression_x);
figure
plt1 = plot(X,logY, 'o');
title('Problem 2a - Exclusive Scan Timing Results');
xlim([5,12])
xlabel('Log2(#Integers)');
ylabel('Log10(Merge Sort Time [ms])');
hold on
plt2 = plot(regression_x, regression_y);
legend([plt1, plt2], {'Actual' 'Regression'});
hold off