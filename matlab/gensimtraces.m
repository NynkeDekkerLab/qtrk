function gensimtraces()

    mag = 50;
    
    %fixluterr = simulate(mag, 1, 0);
    %bmluterr = simulate(mag, 

    plot ( [20:10:200], beadcount ( [20:10:200]));
    
end

function numbeads = beadcount(mag)
    magfactor = 0.5.^(mag / 50 - 1);
    % good yield per 2000x2000 view in falcon2
    yield = 200;
    numbeads = magfactor * yield;
end

function errors=simulate(mag, fixlut, makeplot)
    %simpath='D:\jcnossen1\cudaqi-tracker\qtrk\Debug\cudatrackd.exe';
    simpath='D:\jcnossen1\cudaqi-tracker\qtrk\Release\cudatrack.exe';

    N = 500;
    config.roi = 100;
    config.qi_iterations = 3;
    config.qi_minradius = 8;
    config.zlut_minradius = 2;
    config.zlut_radial_coverage = 3;
    config.zlut_angular_coverage = 0.7;
    config.zlut_roi_coverage = 1;
    config.qi_radial_coverage = 1.5;
    config.qi_angular_coverage = 0.7;
    config.qi_roi_coverage = 0.8;
    config.qi_angstep_factor = 1.5;
    config.lutsmpfile = [cd '\\lutsmp.jpg'];
    config.epb = 15.768658; % falcon2 level
    config.cuda = 0;
    
    lutfile = [cd '\\refbeadlutpos\\traces\\lut\\lut000.png'];
    lut = imread(lutfile);
    luth = size(lut,2);
    
    if fixlut
        fixlutfile = replace_ext(lutfile,'.jpg');
        imwrite(lut, fixlutfile, 'Quality', 100);

        config.fixlut=fixlutfile;
    end
    
    pos = repmat([config.roi/2 config.roi/2 luth/2],N,1) + rand;
    %pos = rand(N, 3) .* repmat([1 1 1], N,1) + repmat([config.roi/2 config.roi/2 luth/2],N,1);
    results = runsim(pos, config, simpath); 
    
    if makeplot
        figure; plot( [ results(:,1) pos(:,1) ] );
        figure; plot( [ results(:,3) pos(:,3) ] );
    end
    
    errors = results-pos;
    fprintf('X: std: %f Z: std: %f\n', std(errors(:,1)), std(errors(:,3)));
end


function newfn= replace_ext(fn, newext)

    [path, name]=fileparts(fn);
    newfn = [path name newext];

end


