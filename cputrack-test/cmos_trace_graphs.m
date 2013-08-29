function cmos_trace_graphs()

    ofs_gain_stdev = dlmread('offset_gain_stdev.txt');

    for k=0:4
        figure(k+1);
        tr = dlmread(sprintf('noiselev%d\\trace.txt',k));
        d = tr(:,3);
        d = (d-d(1))*100;
        plot([d smooth(d, 200)]); % plot Z
        title(sprintf('Gain (mean, stdev): (1,%.2f)', ofs_gain_stdev(k+1,2)));
        ylabel('Z Position (nm)');
        xlabel('Frame');
        
    end

end