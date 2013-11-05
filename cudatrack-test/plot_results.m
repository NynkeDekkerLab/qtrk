function plot_results()

r=dlmread('benchmark-results.txt');
r=r(3:end,:);

PixelSize = 146; % nm
StepSize = 50; % nm
xacc = r(:,2) * PixelSize;
zacc = r(:,4) * StepSize;

%plotyy(r(:,1),[ r(:,5) r(:,7) ], r(:,1), r(:,8) );
subplot(211);
plot(r(:,1),[ xacc zacc ]);
ylabel('Accuracy (st.dev) [nm]');
xlabel('Region-of-interest size [pixels]');
legend('X', 'Z');
subplot(212);
plot(r(:,1), r(:,8));
ylabel('Localization speed (img/s)');
xlabel('Region-of-interest size [pixels]');
legend('Intel Xeon E5-1650');

title('Simulated accuracy of localization');

end