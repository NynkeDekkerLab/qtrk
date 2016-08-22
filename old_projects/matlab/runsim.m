function outputpos = runsim(inputpos, config, path)

    if (nargin<3)
        path = 'cudatrack.exe';
    end

	%check_strarg(args, "output", &outputfile);
	%check_strarg(args, "fixlut", &fixlutfile);
	%check_strarg(args, "inputpos", &inputposfile);    
    
    config.inputpos = tempname;
    dlmwrite(config.inputpos, inputpos, '\t');
    
    config.output = tempname;

    cmdline = '';
    names=fieldnames(config);
    for k=1:length(names)
        cmdline = [ cmdline ' ' names{k} ' ' num2str(config.(names{k}))];
    end
    
    cmdline = [path ' ' cmdline];
 
    fprintf('cmdline: %s\n',cmdline);
    system(cmdline);
    
    outputpos = dlmread(config.output, '\t');
    
end

