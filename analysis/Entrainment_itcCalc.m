%% Calculate TFR and ITC for one channel. Used for Oscillation data analysis
% for all channels - use EntrainmentItpc (takes time). 
% Run after MAIN_Analysis_Entrainment (needs only DATA). 

refDat =  rerefData(DATA,37);
DATA_r = refDat;
clear DATA
%%
clear itcAll dataTemp spectAll spectrumEst itc Group data DATA_I itc sumItc
% calculate TFR and ITC
tic
Group = 9+12; %1-TD/13-ASD

chnI = 1;
gwidthWlt = 3; %Should be > 3
freqoi = 1:20; % for long trials and delta/theta: 0.5:0.5:13; 
widthWlt = linspace(3,5,30); %recommended 3 as minimum
CHAN = 'POz';  %CHAN = 'Cz'; %calc on POz for Figure 4 (was plotted with Oz for veryLong)
chns = find(strcmp(ERP{1}{1}.label,CHAN)); % enter the ID of channels to analyze
LENGTH = size(ERP{1}{1}.avg,2);
TIMEBEFORE_STIM = 1; 
%numTrl = 1; % enter the number of trials
t = [1:LENGTH]/256-TIMEBEFORE_STIM;
 
% calculate ITPC
% try one to initialize size of timeoi and freqoi
% data is a fieldtrip structure with single trial data
% inputs: 
% dataTemp - single trial data matrix (chans X time)
% t - time vector in seconds



dataTemp = DATA_r{Group}{1}{1}(chns,:);
[~,freqoi,timeoi] = ft_specest_wavelet(dataTemp, t, 'freqoi', freqoi,'width', widthWlt, 'gwidth',gwidthWlt);
 
 
for k = 1:length(DATA_r{Group})
    for l = 1:length(DATA_r{Group}{k})
        DATA_I{k}(:,:,l) = DATA_r{Group}{k}{l}; %concat trials as 3rd element
    end
end
 %%
 for k =  1:length(DATA_I) % participants
% initialize matrix for ITC
% initialize matrix for spectrogram
    % this matrix holds the instantaneous phase for each trial, frequency
    % and time point 
    numTrl = size(DATA_I{k},3);
    spectAll = zeros(numTrl,length(freqoi),length(timeoi));
    for trlI = 1 : numTrl
        % select the correct trial and channel
        dataTemp = DATA_I{k}(chns(chnI),:,trlI);
        % calculate time frequency analysis using wavelets
        [spectrumEst,freqoi,timeoi] = ft_specest_wavelet(dataTemp, t, 'freqoi', freqoi, 'width', widthWlt, 'gwidth',gwidthWlt);
        spectAll(trlI,:,:) = spectrumEst(1,:,:);
    end
    % calculate itc for a single channel across trials
    itc{k}(chnI,:,:) = it_calcITC(spectAll);
 end
 
 sumItc = zeros(length(freqoi),LENGTH);
for k = 1:length(itc)
    sqItc = squeeze(itc{k});
    sumItc = sumItc+sqItc;
end
toc

%%

if Group == 9+12
    TdGroup = Group;
    itcAllTd = sumItc./length(DATA_r{Group});
    data = itcAllTd;
    itcTD = cell2mat(itc'); %concatenated subjects 
elseif Group == 11
    itcAllTd_2 = sumItc./length(DATA_r{Group});
    data = itcAllTd_2;
    itcTD_2 = cell2mat(itc'); %concatenated subjects 
elseif Group == 13
    ASDGroup = Group;
    itcAllAsd = sumItc./length(DATA_r{Group});
    data = itcAllAsd;
    itcASD = cell2mat(itc');
elseif Group == 23
    itcAllAsd_2 = sumItc./length(DATA_r{Group});
    data = itcAllAsd_2;
    itcASD_2 = cell2mat(itc'); %concatenated subjects 
end
 
%%
 data = itcAllAsd - itcAllAsd_2;
 
 %%
 data = itcAllTd - itcAllTd_2;

%% to plot the difference between the groups in Cue condition
data = itcAllTd - itcAllAsd; 

%%
data = itcAllTd_2 - itcAllAsd_2; 

%%
data = itcAllTd_2 - itcAllAsd_2 - (itcAllTd - itcAllAsd);

%% stats for groups/individuals

itcTD_ISO = nanmean(itcTD,3);                %average across trials
itcTD_Jitter = nanmean(itcTD_2,3);


itcASD_ISO = nanmean(itcASD,3);                %average across trials
itcASD_Jitter = nanmean(itcASD_2,3);



