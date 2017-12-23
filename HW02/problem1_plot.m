X = [10 11 12 13 14 15 16 17 18 19];
Y1 = [0.135393 0.265668 0.564282 1.161096 2.401104 5.097622 10.644817 18.273104 38.197522 79.060030];
Y2 = [0.128351 0.207022 0.396448 0.834399 1.699739 3.500590 7.377765 15.482033 32.592861 68.535294];
logY1 = log10(Y1);
logY2 = log10(Y2);
p1 = polyfit(X, logY1, 1);
p2 = polyfit(X, logY2, 1);
regression_x = linspace(10,19);
regression_y1 = polyval(p1,regression_x);
regression_y2 = polyval(p2,regression_x);

figure
subplot(2,2,1);
plt1 = plot(X, logY1, 'o');
title('Problem 1a - Mergesort Time Consumed');
xlim([10,19])
ylim([-5, 5])
xlabel('Log2(#Integers)');
ylabel('Log10(Sorting Time [ms])');
hold on
plt2 = plot(regression_x, regression_y1);
legend([plt1, plt2], {'Actual' 'Regression'});
hold off

subplot(2,2,2);
plt3 = plot(X,logY2, '*');
legend('Actual');
title('Problem 1b - Qsort Time Consumed');
xlim([10,19])
ylim([-5, 5])
xlabel('Log2(#Integers)');
ylabel('Log10(Sorting Time [ms])');
hold on
plt4 = plot(regression_x, regression_y2);
legend([plt3, plt4], {'Actual' 'Regression'});
hold off

subplot(2,2,3);
plt5 = plot(X,logY1, 'o');
legend('Problem1a');
title('Sorting Timing Comparison');
xlim([10,19])
ylim([-5, 5])
xlabel('Log2(#Integers)');
ylabel('Log10(Sorting Time [ms])');
hold on
plt6 = plot(X,logY2, '*');
legend([plt5, plt6], {'Problem1a' 'Problem1b'});
hold off
