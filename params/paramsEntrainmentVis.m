%This code writes variables of fixed parameters to be used at pre
% processing of EEG experiments. 

function params = paramsEntrainmentVis
params.refchan = 'AFz';
params.eventvalue = [31,32,34,39,40,41,99]; %
params.restingstate = [200,400];
params.retingstateTime = [10000,30000]; %resting state trials (in ms)
params.numelec = 64;
params.eventtype  = 'STATUS';
params.prestim = 0.5;  %     % (seconds)  
%params.poststim = 4; %poststim is variable, set in the trialfun (params.trialfun = 'all_triggers_entrain').
params.offset = 0;
params.powerline = 'yes';
% params.channel = []; 
params.continuous = 'yes';
params.makelayout = 0;
params.layout = '64_lay.mat';
params.yaxis = [-2000 2000];
params.resample = 256;
params.numcomponent = 20;
params.dftfreq = 60;
params.lpfreq = 45; % to check entrainment for the 1.5 hz; %5; 
params.demean = 'yes';
params.hpfilter = 'yes';
params.hpfreq = 0.1; 
params.hpfiltord = 5;
params.bpfreq = [1.4 1.6];
params.thres = 80; %(microvolts)
params.detrend = 'yes'; 
params.basicRhythm = 0.666;
params.basicTime = 5;
params.shortTime = 2.750;
params.longTime = 8;
params.timeOrder = [5000,8000,2750,5000,5000,5000,5000];
params.reref = 'no';
params.trialfun = 'all_triggers_entrain';
params.refchannel = 'AFz';
params.baselinewindow = [0 0.1];
params.basicColors =  [[0,0,0];[212,32,143]./255]; %pinks and greys (modified for the CNS poster, March 2018).

