function plot_speeds()
    s690=dlmread('speeds-geforce690.txt');
    s560=dlmread('speeds-geforce560.txt');
    s570=dlmread('speeds-geforce570.txt');
    s570po2=dlmread('speeds-po2.txt');
    s570=dlmread('speeds.txt');
    
    %s = [ s690(:,[1 3]) s560(:,3) s570(:,3) ];
    s = s570(:,[1 2 3]);

    figure();
    x = 40 + ((0:23)*5);
    plot(x,s, 'LineWidth', 2);
    set(gca,'FontSize', 15);
    title('Tracking speed vs ROI size');
    xlabel('ROI size [pixels]');
    
    %legend( '12 core CPU', 'Geforce 690', 'Geforce 560', '4 core cpu', 'Geforce 560 (bad parallel code)');
    legend( 'CPU Intel i5-2400', 'Geforce570 (mem)', 'Geforce570 (tex)' );
