% This code writes variables of fixed parameters to be used at pre
% processing of EEG experiments. 

function params = paramsERPVisLong
params.trialfun = 'all_triggers_ERP_RT';
params.eventvalue = [31,32,33,34,35,36,37,38,39,40,41,99]+11; %
params.restingstate = [200,400];
params.retingstateTime = [10000,30000]; %resting state trials (in ms)
params.numelec = 64;
params.eventtype  = 'STATUS';
% params.prestim = 0.5;  %     % (seconds)  
% params.poststim = 4; 
params.prestim = 3;    % (seconds)  
params.poststim = 3; 
params.offset = 0;
params.powerline = 'yes';
% params.channel = []; 
params.continuous = 'yes';
params.makelayout = 0;
params.layout = '64_lay.mat';
%params.layout = '160_lay.mat';
params.yaxis = [-2000 2000];
params.resample = 256;
params.numcomponent = 20;
params.dftfreq = 60;
params.lpfreq = 55; % to check entrainment for the 1.5 hz; %5; 
params.demean = 'yes';
params.detrend = 'yes';
params.hpfilter = 'yes';
params.hpfreq = 0.11; 
params.hpfiltord = 5;
%params.bpfreq = [1.4 1.6];
params.thres = 100; %(microvolts)
params.baselinewindow = [0.1 0.2];
params.basicRhythm = 0.666;
params.basicTime = 5;
params.shortTime = 2.750;
params.longTime = 8;
params.reref = 'yes';
params.refchannel = 'AFz'
params.x = [-params.prestim:1/256:params.poststim]; 
params.basicColors =  [[0,0,0];[212,32,143]./255]; %pinks and greys (modified for the CNS poster, March 2018).

