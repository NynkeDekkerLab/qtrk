function plot_results()


    figure(1);
    plot_roi_sizes();

    figure(2);
    plot_qirad();

end

function plot_roi_sizes()
    r=dlmread('roi-sizes.txt');
    r=r(3:end,:);

    PixelSize = 146; % nm
    StepSize = 50; % nm
    xacc = r(:,2) * PixelSize;
    xbias = r(:,5) * PixelSize;
    zacc = r(:,4) * StepSize;
    zbias = r(:,7) * StepSize;
    %plotyy(r(:,1),[ r(:,5) r(:,7) ], r(:,1), r(:,8) );
    subplot(211);
    p=plot(r(:,1),[ xacc zacc xbias zbias ]);
    set(p(1),'LineStyle','--');
    set(p(2),'LineStyle','--');
    
    legend('X Scatter', 'Z Scatter', 'X Bias', 'Z Bias');
    ylabel('Accuracy (st.dev) [nm]');
    xlabel('Region-of-interest size [pixels]');
    subplot(212);
    plot(r(:,1), r(:,8));
    ylabel('Localization speed (img/s)');
    xlabel('Region-of-interest size [pixels]');
    %legend('Intel Xeon E5-1650');
    legend('GTX 580');

    title('Simulated accuracy of localization');

end

function plot_qirad()

    r=dlmread('qiradstepdens.txt');
    r=r(3:end,:);

    PixelSize = 146; % nm
    StepSize = 50; % nm
    xacc = r(:,2) * PixelSize;
    xbias = r(:,5) * PixelSize;
    zacc = r(:,4) * StepSize;
    zbias = r(:,7) * StepSize;

    %plotyy(r(:,1),[ r(:,5) r(:,7) ], r(:,1), r(:,8) );
    subplot(211);
    semilogy(r(:,1),[ xacc zacc xbias zbias ]);
    legend('X Scatter', 'Z Scatter', 'X Bias', 'Z Scatter');
    ylabel('Accuracy (st.dev) [nm]');
    xlabel('QI Radial Step Density (smp/pixel length)');
    legend('X Scatter', 'Z Scatter', 'X Bias', 'Z Scatter');
    subplot(212);
    plot(r(:,1), r(:,8));
    ylabel('Localization speed (img/s)');
    xlabel('Region-of-interest size (smp/pixel length)');
    %legend('Intel Xeon E5-1650');
    legend('GTX 580');

    title('Simulated accuracy of localization');

end
