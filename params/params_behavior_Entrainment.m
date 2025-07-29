function params = params_behavior_Entrainment

params.sr = 512;
params.labels = {'TD','ASD'};
%params.groups = [1,3]; %groupCodes
params.id_clin = 3;
params.loc = 2;
params.triggerCodeVis = [31:41,99]+11;
params.triggerCodeAud = [1:11,69]+11;
params.exclude = [10158,12386]; %subs to exclude (onw raw for each stim)
%params.catch = [8,18];
params.responseCode = 100;
%params.tolerance = [80 1500]; %ms
%params.otherparams = paramsVAMP_aud_short;
params.colors = [[0,0,0];[122,122,122]./255;[212,32,143]./255;[170,125,154]./255]; 
end


