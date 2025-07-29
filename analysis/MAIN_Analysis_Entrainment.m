 
clear;
prompt1 = 'At Work? (1-Y / 2-N)? ';  
location = input(prompt1); %1 for lab, 2 for laptop
prompt2 = 'Visual (1) / Auditory (2)? ';
sense = input(prompt2); %1 for visual, 2 for auditory
prompt3 = 'Long broad-band (1) or Low-passed (2) ERP (3) CNV (4) CNV unique (5) ERP long (for ITPC, and gradation calc) (6) Resting State (7) or Phase-RT (8)? ';
Signal = input(prompt3); %analysis on 1 - broadband eeg or 2- 5hz lowpassed filtered eeg; 6 (erp long - for the time-freq itpc).
 
if location == 1 && sense == 1 && Signal ==3 
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/EntrainProcessed_ERP vis'); %to test each sub individually
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 1 && sense == 1 && Signal == 7
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/Resting State'); %to test each sub individually
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 2 && sense == 1 && Signal == 7 
    PATH_data = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/Resting State'); %for mac directory
    PATH_analysis = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 2 && sense == 1 && Signal == 6 
    PATH_data = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/EntrainProcessed_ERPLong vis'); %for mac directory
    PATH_analysis = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 1 && sense == 1 && Signal == 6 
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/EntrainProcessed_ERPLong vis'); %to test each sub individually
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 1 && sense == 1 && Signal == 1
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP long vis'); %for pc directory
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
        startup;
elseif location == 1 && sense == 2 && Signal == 1
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP long aud'); %for pc directory
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
        startup;
elseif location == 2 && sense == 1 && Signal ==3 
    PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/EntrainProcessed_ERP vis'); %for mac directory
    PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Shared with shlomitbeker/Analysis');
    cd('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    startup;
 elseif location == 2 && sense == 2 && Signal ==3 
    PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP Aud'); %for mac directory
    PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Shared with shlomitbeker/Analysis');
    cd('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    startup;
 elseif location == 2 && sense == 1 && Signal ==1 
    PATH_data = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP long vis'); %for pc directory
    PATH_analysis = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup; 
%     PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP long vis'); %for mac directory
%     PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Shared with shlomitbeker/Analysis');
%     cd('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
%     startup;
 elseif location == 2 && sense == 1 && Signal ==4 
    PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/CNV Vis'); %for mac directory
    PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    startup;
elseif location == 2 && sense == 1 && Signal ==5 
    PATH_data = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/CNV_Vis_unique'); %to test each sub individually
    PATH_analysis = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
 elseif location == 1 && sense == 1 && Signal ==4 
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/CNV Vis'); %for mac directory
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('C:\Users\sbeker\Dropbox (EinsteinMed)\Shared with shlomitbeker\GENERAL ANALYSIS FILES');
    startup; 
elseif location == 1 && sense == 1 && Signal ==5 
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/CNV_Vis_unique'); %for mac directory
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
elseif location == 1 && sense == 2 && Signal ==3 
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP Aud'); %for mac directory
    %PATH_data = ('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Processed\ERP Aud\New folder');
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;

elseif location == 2 && sense == 2 && Signal == 6
    PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP long aud'); %for mac directory
    %PATH_data = ('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Processed\ERP Aud\New folder');
    PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Shared with shlomitbeker/Analysis');
    cd('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    startup;

elseif location == 2 && sense == 1 && Signal == 8
     PATH_data = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP_long_RT'); %for mac directory
    PATH_data = ('/Users/shlomit/EinsteinMed Dropbox/Shlomit Beker/Shared with shlomitbeker/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP_long_RT');
     PATH_analysis = ('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
%     PATH_data = ('/Users/shlomit/Dropbox (EinsteinMed)/Shared with shlomitbeker/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP_long_RT'); %for mac directory
%     PATH_analysis =  ('/Users/shlomit/Dropbox (EinsteinMed)/Shared with shlomitbeker/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
%     cd('/Users/shlomit/Dropbox (EinsteinMed)/Shared with shlomitbeker/GENERAL ANALYSIS FILES');
%     startup;

elseif location == 1 && sense == 1 && Signal == 8
    PATH_data = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERP_long_RT'); %for mac directory
    PATH_analysis = ('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
    startup;
end


addpath('/Users/shlomit/Dropbox (EinsteinMed)/Shared with shlomitbeker/GENERAL ANALYSIS FILES');
addpath('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');
addpath('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/GENERAL ANALYSIS FILES');

addpath(PATH_analysis);
cd(PATH_analysis)

if sense == 1 && Signal == 3
    PARAMS = paramsERPVis;
elseif sense == 1 && Signal == 6    
    PARAMS = paramsERPVisLong;
elseif sense == 1 && Signal == 1
    PARAMS = paramsEntrainmentVis;
elseif sense == 2 
    PARAMS = paramsERPAud;
elseif sense == 1 && Signal == 4
    PARAMS = paramsCNV_Vis;
elseif sense == 1 && Signal == 5
    PARAMS = paramsCNV_Vis_unique;
elseif sense == 1 && Signal == 7
    PARAMS = paramsRestState;
elseif sense == 1 && Signal == 8    
    PARAMS = paramsERPVisLong_RT;
end

%PATH_data = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND
%OSCILLATION IN ASD/Processed/Resting
%State/ProessedRestState_usedInPaperAnalysis'; Resting state regular (not
%long)


cd(PATH_data);


%%
if Signal == 3
    analyze = 'EEGVis.mat';
    loadPath = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Data Structures';
    loadPath = 'C:\Users\sbeker\Dropbox (EinsteinMed)\Shared with shlomitbeker\A ENTRAINMENT AND OSCILLATION IN ASD\Data Structures';
elseif Signal == 1
     
end


prompt4 = 'run all files? '; % 1: to run the subjects; 2: to upload the ERPb file
analysis = input(prompt4); 
if analysis==2
    load([loadPath,'/',analyze])

else
    
    
    
    FS = 256;
    matFiles = dir('*.mat');
    numfiles = length(matFiles);
    %1:5
    for i = 1:numfiles
        subDATA = matFiles(i).name;
        load(fullfile(PATH_data,subDATA));
        a = regexp(matFiles(i).name,'\d'); % to identify between ASD vs TD, by the number
        subjectID = matFiles(i).name(a);
        %grandData{i}.data = ICAdata;
        ICAdata = dummy;
        grandData{i}.data = ICAdata;
        grandData{i}.name = subjectID;
        
        if subjectID(2) == '1' || subjectID(2) == '8'
            if Signal == 7
                grandData{i}.trialinfo = ICAdata.trialinfo(:,2)+100;
            else
                grandData{i}.trialinfo = ICAdata.trialinfo+100;
            end
        else
            if Signal == 7
                grandData{i}.trialinfo = ICAdata.trialinfo(:,2);
            else
                grandData{i}.trialinfo = ICAdata.trialinfo;
            end
        end
        %          F = find(grandData{i}.trialinfo == 42); %once there was a bug, and 42 was both target of #1, and code for trig of cond #7.
        %          grandData{i}.trialinfo(F) = 99;
        
        if Signal == 8
          t{i} = unique(grandData{i}.trialinfo(:,1))';

        else
          t{i} = unique(grandData{i}.trialinfo)';
        end
    end
    %%
    trigs = unique(cell2mat(t));
    R = rhythms;
    
    % create a different variable for each resting state seperately (after each condition).
    %% ERP by trigger
    
    clear DATA ERP ERPb y badSubject NAMES NAMES1 grandAvg RT x
    trialNumbers = [];
     
    for i = 1:length(trigs) %loop over the conditions
        flag = 0;
        for numSub = 1:length(grandData)
            %delete too small or large RT
            cfg=[];
            if Signal == 8
                F = find(grandData{numSub}.trialinfo(:,1) == trigs(i) & grandData{numSub}.trialinfo(:,2)> 150 & grandData{numSub}.trialinfo(:,2) < 650) %650
                cfg.trials = F;  
            else
                cfg.trials = find(grandData{numSub}.trialinfo == trigs(i));
            end
            if isempty(cfg.trials)~=1
                grandData{numSub}.name
                timelock = ft_timelockanalysis(cfg, grandData{numSub}.data); % average across all ERP within the same subject and same trigger code
                DATA{i}{numSub} = grandData{numSub}.data.trial(cfg.trials);
                if Signal == 8
                    RT{i}{numSub} = [grandData{numSub}.trialinfo(cfg.trials,2),ones(length(cfg.trials),1)*trigs(i)];
                end
                flag = flag+1;
                trialNum = length(cfg.trials);
                trialNumbers(numSub,i) = trialNum;
                
                %Baseline correct
                cfg=[];
                ERP{i}{flag} = timelock;
                
                %ERP{i}{flag} = ft_timelockbaseline(cfg, timelock);
                ERP{i}{flag}.name = grandData{numSub}.name;
                NAMES{i}(flag) = str2num(ERP{i}{flag}.name);
                condSize{i}(flag) = length(DATA{i}{numSub}); %length(ERP{i}{flag}.cfg.previous.trials);
                y{i}(:,flag) = mean(ERP{i}{flag}.avg,1);
                
            end
        end
        emptycells = find(cellfun(@isempty,DATA{i}));
        DATA{i}(emptycells) = [];
        if Signal == 8
            RT{i}(emptycells) =[];
        elseif  Signal == 3
            x{i} = [1:size(ERP{i}{1}.avg,2)]./FS;
        else
            x{i} = [-PARAMS.prestim:1/256:size(ERP{i}{1}.avg,2)./256];
        end
    end
    
    stats = sum(trialNumbers); % number of trial for each trigger code
    %x = [1:size(ERPb{1}{1}.avg,2)]./FS;
    
    
    %%
    ERPb = ERP;
    
    prompt5 = 'detrend? 0-no, 1- yes  '; %ERP-yes
    detrendTime = input(prompt5);
    if detrendTime == 1
        ERPb = detrendData(ERPb);
    end
    
    prompt5 = 'reref to another channel? (0-no, 1-yes) '; %Aud short - 'TP7'; vis - Afz; aud long for entrainment - Afz,
    Ref = input(prompt5);
    if size(PARAMS.refchannel,1) == 1
        %PARAMS.refchannel = {'AFz'};
        %refChan = PARAMS.refchannel;
        refChan = find(strcmp(ERPb{1}{1}.label,PARAMS.refchannel));

    elseif length(PARAMS.refchannel) > 1
            refChan1 = find(strcmp(ERPb{1}{1}.label,PARAMS.refchannel{1}));
            refChan2 = find(strcmp(ERPb{1}{1}.label,PARAMS.refchannel{2}));
            refChan = [refChan1, refChan2];     
    end
    %ERPb = rerefDataERP(ERPb, refChan);
    ERPb = rerefDataERP(ERPb, 'all');

    
    prompt7 = 're-baseline? 0-no, 1- yes  ';
    baselineTime = input(prompt7);
    if baselineTime == 1
        prompt8 = 'insert win in sec  ';  %ERP [0.1 0.2]; ERP aud [0.2 0.25] CNV [0.69 0.7] ERP long [0 1]
        baselineWin = input(prompt8);
        baseline = round(baselineWin*FS)+1;
        ERPb = baselineDataERP(ERPb,baseline);
    end
    
    ERPsize = length(ERPb);
    
    
    
end


%% grand average on all subjects and all trials

channels = {'all'};  
clear grandAvg;     
trigNum = length(trigs);
    cfg = [];
    cfg.channel = channels;
    grandAvg = cell(1, trigNum);
    for j = 1:trigNum
        grandAvg{j} = ft_timelockgrandaverage(cfg, ERPb{j}{1:length(ERPb{j})});
    end
   

    
%% At this stage - for RT-Phase run Entrain_PL_RT_andResting_3.m           ******************
    
%% Number of trials
flag = 0;
if sense == 1
    groups = [1,4,6,7,8,11,13,14];
else
    groups = [1,9,11,12,13,21,23,24];
end
for i = groups
    flag = flag+1;
    for j = 1:length(DATA{i})
        num{flag}(j) = length(DATA{i}{j});

    end
end


%% plot short ERP
if Signal == 3 || sense == 1
%    CHAN = {'O1','Oz','O2'}; 
%    C = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2})),find(strcmp(grandAvg{1}.label,CHAN{3}))];  
    CHAN = {'Oz','Iz'}; 
     %CHAN = {'O2'}; 

    %C = [find(strcmp(grandAvg{1}.label,CHAN{1}))];

    C = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2}))];
elseif Signal == 3 || sense == 2
        %CHAN = {'F1','F2','Fz'}; % for short trials (AEP)
        CHAN = {'C1','C2','Cz'};
         C = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2})),find(strcmp(grandAvg{1}.label,CHAN{3}))];
elseif Signal == 1 || stim == 4
    %CHAN = {'O1','Oz','O2'}; % for entrainment: 'C1','Cz','C2' for visual sequence: 'AF3','Fp1','AFz','Fpz','A7','PO3','POz','PO4',
    CHAN= {'AF3','AF4','AFz'}
    C = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2})),find(strcmp(grandAvg{1}.label,CHAN{3}))];
        %find(strcmp(grandAvg{1}.label,CHAN{4})),find(strcmp(grandAvg{1}.label,CHAN{5})),find(strcmp(grandAvg{1}.label,CHAN{6}))];
end
newERPb = ERPb;
% PLOT all subjects (broadband)
 
if Signal == 3 || Signal == 4 || Signal == 6
     %titles = {'TD VEP','ASD VEP'};
     titles = {'cond 1 666','cond 2 333','cond 2 999','cond 3 250','cond 3 400','cond 3 550','cond 3 700','cond 3 850'...,
         'cond 4 rand' ,'cond 5 pseudo rand','cond 6 jitter small','cond 7 jitter large'};
     
elseif Signal == 1 
    titles = {'cond 1 666','cond 2 333/999','cond 3 increasing'...,
         'cond 4 rand' ,'cond 5 pseudo rand','cond 6 jitter small','cond 7 jitter large'};
elseif Signal == 5
     titles = {'33','39','40','41','99'} 
elseif Signal == 7
    titles = {'31','39','41','99','131','139','141','199'};
    newERPb = [ERPb(1),ERPb(4),ERPb(6),ERPb(7),ERPb(9),ERPb(13),ERPb(15),ERPb(16)];
end
fig1 = figure;
for i = 1:24
    x{i}= [-PARAMS.prestim:1/256:PARAMS.poststim]; 
    
end

% plot and get "average" - the averages for group and condition
average = plotMeanAverage(fig1,trigNum,x,newERPb,C,titles,0,0); %before last - de-trend (1) or not (0). last entry is pause (0/1);

%% run FFT on the 5.5sec segments (Fig 2 in the paper): plotEntrainmentFFT.m


%% Entrainment: plot 3 columns on 7 conditions. Columns: wideband; low-pass filter; fft
condOrd = repmat(1:7,1,2);
Ylim = [-10 5];
cfg = [];
cfg.layout = '64_lay.mat';
cfg.interactive = 'yes';
colors = rand(7,3);
colors = cat(1,colors,colors,[1,1,1]);
colors = {'k','r'};
cfg.channel = {'C1','C2','Cz'};
% Plot wideband EEG 
for i = 1:length(grandAvg)

    if i < length(grandAvg)/2+1
        figure(1);
    else 
        figure(2);
    end
     subplot(length(grandAvg)/2,1,condOrd(i)); hold on;
     %ADD COND NAME IN THE CORNER title(num2str(trigsAnal(i)));
     if i<length(grandAvg)/2+1
         cfg.graphcolor = colors{1};
     else
         cfg.graphcolor = colors{2};
     end
      cfg.linewidth = 2; 
      cfg.ylim = [-10 7];
      cfg.figure = 'no'; %plot on the same figure
%       cfg.figure = 'yes';
%       cfg.figurename = 'Figure 1';
      ft_singleplotER(cfg,grandAvg{i});  
      hold on;
    for j = 1:length(R{condOrd(i)})-1
        line([R{condOrd(i)}(j)  R{condOrd(i)}(j)],Ylim,'Color','k');
    end       
end

%% lowpass and plot
cfg = [];
cfg.continuous = 'yes';
cfg.lpfilter = 'yes';
%cfg.hpfilter = 'yes';
cfg.lpfreq = 1.7;
%cfg.hpfreq = 5;
cfg.hpfiltord = 3;
cfg.lpfilttype = 'fir'

for i = 1:length(ERPb)
    for j = 1:length(ERPb{i})
        ERPfilt{i}{j} = ft_preprocessing(cfg,ERPb{i}{j});  
    end
end

% gradAvg
channels = {'all'};  
clear grandAvg;     
    cfg = [];
    cfg.channel = channels;
    grandAvgfilt = cell(1, trigNum);
    for j = 1:trigNum
        grandAvgfilt{j} = ft_timelockgrandaverage(cfg, ERPfilt{j}{1:length(ERPfilt{j})});
    end
% Filt the DATA structure
LPF = designfilt('lowpassfir', 'PassbandFrequency', 1.7, 'StopbandFrequency', 4, 'PassbandRipple', 1, 'StopbandAttenuation', 60, 'SampleRate', 256);

for i = 1:length(DATA)
    for j = 1:length(DATA{i})
        for k = 1:length(DATA{i}{j})
            DATAfilt{i}{j}(k,:) = filtfilt(LPF,DATA{i}{j}{k}(64,:)); 
        end
    end
end

%plot the low-passed filt data
condOrd = repmat(1:7,1,2);
Ylim = [-10 5];
cfg = [];
cfg.layout = '64_lay.mat';
cfg.interactive = 'yes';

colors = {'k','r'};
cfg.channel = {'O1','O2','Oz'};
% Plot lowpass EEG 
figure
for i = 1:length(grandAvgfilt)
     subplot(length(grandAvgfilt)/2,1,condOrd(i)); hold on;
     if i<length(grandAvgfilt)/2+1
         cfg.graphcolor = colors{1};
     else
         cfg.graphcolor = colors{2};
     end
      cfg.linewidth = 2; 
      cfg.ylim = [-10 7];
     cfg.figure = 'no';
      hold on;
      ft_singleplotER(cfg,grandAvgfilt{i});        
    for j = 1:length(R{condOrd(i)})-1
        line([R{condOrd(i)}(j)  R{condOrd(i)}(j)],Ylim,'Color','k');
    end       
end



%% plot the ERPs over selected sensors

cfg = [];
cfg.layout = '64_lay.mat';
cfg.interactive = 'yes';
y = [-15, 15];
colors = rand(7,3);
colors = cat(1,colors,colors,[1,1,1]);

if sense == 1
    cfg.channel = {'O1','O2','Oz'};
     %cfg.channel = {'AF1','AF4','AFz'};
elseif sense == 2
    cfg.channel = {'F1','F2','Fz'};
end
if Signal == 3 || Signal == 4
    condOrd = [1:12,1:12];
elseif Signal == 3
    condOrd = [1:7,1:7];
end

switch Signal
    
    case 1      %broad band filter.
        
        for i = 1:length(grandAvg)
            
            if i < length(grandAvg)/2+1
                figure(1);
            else 
                figure(2);
            end
            hold on; 
            subplot(length(grandAvg)/2,2,condOrd(i));
            %ADD COND NAME IN THE CORNER title(num2str(trigsAnal(i)));
            cfg.linewidth = 2; cfg.graphcolor = colors(i,:);
            cfg.ylim = [-5 8];
            ft_singleplotER(cfg,grandAvg{i}); 

            for j = 1:length(R{condOrd(i)})-1
                line([R{condOrd(i)}(j)  R{condOrd(i)}(j)],y,'Color','k');
            end
            
        end
    case 3
        figure;
        sgtitle('VEP for each condition')
        colors = PARAMS.basicColors;
         figure;
        sgtitle('VEP for each condition')
        colors = PARAMS.basicColors;
        figid = [1:12];%[1:24];
         for i = 1:length(grandAvg)/2
            hold on; 
            subplot(4,3,figid(i));
            cfg.linewidth = 2; 
            cfg.graphcolor = [colors(1,:);colors(2,:)];
            ft_singleplotER(cfg,grandAvg{i},grandAvg{i+12});
            xt = get(gca, 'XTick');
            set(gca, 'XTick',xt, 'XTickLabel',xt-0.2)
            ylim([-6 8]);
            Line = line([0.2 0.2],[-6 8],'Color','k');
            title(titles(i))
         end
            hold on;
            xt = get(gca, 'XTick');
            set(gca, 'XTick',xt, 'XTickLabel',xt-0.2)
            ylim([-6 8]);
            Line = line([0.2 0.2],[-6 8],'Color','k');
         annotation('textbox', [0, 0.8, 0, 0], 'string', 'TD')
         annotation('textbox', [0, 0.4, 0, 0], 'string', 'ASD')

    case 2      %low-passed filter. Plot with FFT.
         figure
        for i = 1:length(grandAvg)
            figid = [1,3,5,7,9,11,13];
            hold on; subplot(7,2,figid(i));
            cfg.linewidth = 2; cfg.graphcolor = colors(i,:);
            ft_singleplotER(cfg,grandAvg{i});
            
            hold on;
            for j = 1:length(R{i})
                line([R{i}(j)  R{i}(j)],y,'Color','k');
            end
            
            title(num2str(i));
            hold on;
        end
        
        % spectralAnal
        data = cell(1, length(grandAvg));
        for i = 1:length(grandAvg)
            data{i} = grandAvg{i}.avg;
        end
        
        elec = [find(strcmp(grandAvg{1}.label, cfg.channel{1})),find(strcmp(grandAvg{1}.label, cfg.channel{2}))...
            find(strcmp(grandAvg{1}.label, cfg.channel{3}))];
        %elec = [find(strcmp(grandAvg{1}.label, 'CP1')),find(strcmp(grandAvg{1}.label, 'CP2'))...
        % find(strcmp(grandAvg{1}.label, 'CPz'))];
        [f, entrainSpec] = spectralAnal(data,FS,elec,3,1); %refers to: Data, FS, electrodes, casenum(1-3),plotting (yes-1, no-0).
        for i = 1:length(entrainSpec)
            figid = [2,4,6,8,10,12,14];
            hold on; subplot(7,2,figid(i));
            plot(f, entrainSpec{i}, 'Color',colors(i,:),'LineWidth',2);
            ylim([0 0.35]);
            set(gca,'fontsize',10)
            hold on;
        end
        figure;
        run = 0;
        for i = [1,4,6]
            run=run+1;
            subplot(3,1,run)
            cfg.linewidth = 2; cfg.graphcolor = colors(i,:); cfg.layout = '64_lay.mat';cfg.zlim = [-0.9 0.9];
            
            ft_singleplotER(cfg,grandAvg{i});
            hold on;
            set(gca,'fontsize',20)
            for j = 1:length(R{i})
                line([R{i}(j)  R{i}(j)],y,'Color','k');
            end
        end
        figure;
        run = 0;
        for i = [1,4,6]
            run=run+1;
            plot(f, entrainSpec{i},'Color',colors(i,:),'LineWidth',2);
            hold on
            set(gca,'fontsize',20);
        end
end

%% Stats for ERPs. C is as defined before
clear maxComp
%win = round(0.1*256+[0.17 0.3]*256);
win = round(0.2*256+[0.12 0.16]*256);

% CHAN = {'O1','Iz'};

% C = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2})),find(strcmp(grandAvg{1}.label,CHAN{3}))];
    ERPbSel = [ERPb(1),ERPb(9),ERPb(11),ERPb(12),ERPb(13),ERPb(21),ERPb(23),ERPb(24)];


for i = 1:length(ERPbSel)
    %figure
    for j = 1:length(ERPbSel{i})
        %for k = 1:length(DATAselect{i}{j})
            %plot(mean(ERPbSel{i}{j}.avg(C,:)))
            hold on; 
            %maxComp{i}(j,1) = mean(max(ERPbSel{i}{j}.avg(C,win(1):win(2)),[],2));  %max  comp
            maxComp{i}(j,1) = std(mean(ERPbSel{i}{j}.avg(C,win(1):win(2))));  %max  comp, mean across trials

            %maxComp{i}(j,1) = mean(mean(ERPbSel{i}{j}.avg(C,win(1):win(2))));  %max  comp, mean across trials

        %end
    end
end

% print VEP results
meansVEP = [mean(cellfun(@mean, maxComp(1:4))) mean(cellfun(@mean, maxComp(5:8)))]
semVEP = [mean(cellfun(@std, maxComp(1:4)))./sqrt(20) mean(cellfun(@std, maxComp(5:8))./sqrt(17))]

% Anova
Y = cat(1,maxComp{1}, maxComp{2},maxComp{3}, maxComp{4},maxComp{5}, maxComp{6},maxComp{7},maxComp{8}); 
g1 = [ones(1,length(maxComp{1})),ones(1,length(maxComp{2})),ones(1,length(maxComp{3})),ones(1,length(maxComp{4})),...
    ones(1,length(maxComp{5}))*2,ones(1,length(maxComp{6}))*2,ones(1,length(maxComp{7}))*2,ones(1,length(maxComp{8}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(maxComp{1})),ones(1,length(maxComp{2}))*2,ones(1,length(maxComp{3}))*3,ones(1,length(maxComp{4}))*4,...
    ones(1,length(maxComp{5})),ones(1,length(maxComp{6}))*2,ones(1,length(maxComp{7}))*3,ones(1,length(maxComp{8}))*4]; %condition 1-ISO 2-jitS 3-jitL 4 - rand
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

figure;
st_boxdotplot([1:8],{maxComp{1}; maxComp{5};maxComp{2}; maxComp{6}; maxComp{3}; maxComp{7}...
   ;maxComp{4}; maxComp{8}} ,[0,0,0;0.2,0.8,0.5;0.5,0.1,0.9;0.4,0.2,0.2;0,0,0;0.2,0.8,0.5;0.5,0.1,0.9;0.4,0.2,0.2...
   ],'iqr',[],[],[],[],80,[],[],1);

Xaxis = [-0.1:1/256:0.5];

figure; 
subplot(2,1,1)
plot(Xaxis, mean(grandAvg{1}.avg(C,:))); hold on; plot(Xaxis, mean(grandAvg{9}.avg(C,:))); hold on; ...
    plot(Xaxis, mean(grandAvg{11}.avg(C,:))); hold on; plot(Xaxis, mean(grandAvg{12}.avg(C,:)));
ylim([-4 6])
xlim([0 0.5])
subplot(2,1,2)
plot(Xaxis, mean(grandAvg{13}.avg(C,:))); hold on; plot(Xaxis, mean(grandAvg{21}.avg(C,:))); hold on; ...
    plot(Xaxis, mean(grandAvg{23}.avg(C,:))); hold on; plot(Xaxis, mean(grandAvg{24}.avg(C,:)));
ylim([-4 6])
xlim([0 0.5])

%% Names for ERPsel

for i = 1:length(ERPbSel)
    for j = 1:length(ERPbSel{i})
        VEP_IDs{i}(:,j) = str2num(ERPbSel{i}{j}.name);
    end
end

%%  plot ERP in interactive mode

cfg = [];
cfg.linewidth = 1;
cfg.layout = '64_lay.mat';
cfg.interactive = 'yes';
cfg.showscale = 'no';
cfg.showlabels = 'yes';
%cfg.ylim = [-1 -1];

Cond = 1
figure; ft_multiplotER(cfg,grandAvg{1},grandAvg{13}); %cfg.graphcolor = 'bg';

figure; ft_multiplotER(cfg,grandAvg{1},grandAvg{11},grandAvg{12},grandAvg{9}); %cfg.graphcolor = 'bg';
figure; ft_multiplotER(cfg,grandAvg{1+12},grandAvg{11+12},grandAvg{12+12},grandAvg{9+12}); %cfg.graphcolor = 'bg';

figure; ft_multiplotER(cfg,grandAvg{4+12},grandAvg{5+12},grandAvg{6+12},grandAvg{7+12}); %cfg.graphcolor = 'bg';

figure; ft_multiplotER(cfg,grandAvg{1},grandAvg{1+12}); %cfg.graphcolor = 'bg';
figure; ft_multiplotER(cfg,grandAvg{7},grandAvg{7+12}); %cfg.graphcolor = 'bg';


figure; ft_multiplotER(cfg, ERPb{2}{1:length(ERPb{3})}); %cfg.graphcolor = 'bg';
figure; ft_multiplotER(cfg, ERPb{1+12}{6}); %cfg.graphcolor = 'bg';



figure; ft_multiplotER(cfg, ERPb{13}{15}); %cfg.graphcolor = 'bg';


%% CNV 

% upload structures and call function CNV
cd('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Processed\ERPb CNV structures') 

%cd('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed/ERPb CNV structures');
load ERPb_CNVunique
load ERPb_CNVAll
trigCNV{1} = [31,32,33,34,35,36,37,38,39,40,41,99,131,132,133,134,135,136,137,138,139,140,141,199];
trigCNV{2} = [33,39,40,41,99,133,139,140,141,199];
CHAN = {'Fp1','AF3','Fpz','Fp2','AF4','AFz'}; 
layout = '64_lay.mat';

C = [find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{1})),find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{2})),find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{3}))...
    find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{4})),find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{5})),find(strcmp(ERPb_CNVAll{1}{1}.label,CHAN{6}))];  
conditions = [1,13;1,9;13,21]
window = [180,256];
CNVvolt = CNV(layout, trigCNV,conditions,C ,window,ERPb_CNVAll, ERPb_CNVunique);

%  Plot individual participants for a specific condition (COND)
COND = 22;
ERPb = ERPb_CNVAll;
x = [1:size(ERPb{1}{1}.avg,2)]./256;

figure;
for i = 1:length(ERPb{COND})
  subplot(5,4,i)
  plot(x, mean(ERPb{COND}{i}.avg(C,:),1),'r');
  ylim([-10 10])
  title(ERPb{COND}{i}.name)
  
end


%% Analysis: habituation in repeating stimuli
%reref and rebaseline

load DATA_rerefed;


%% to find the exact indices of the trials 
clear condInd
spec_cond = 1;
for i = 1:length(grandData)
    AAA = diff(find(grandData{i}.trialinfo==spec_cond|grandData{i}.trialinfo==spec_cond+100));
    AAA(AAA>1)=0;
    condInd{i} = find(AAA==0);
end

%% get IDs and first trials, to measure habituation on the first block. 
for i = 1:length(grandData)
    nameTrials(i,:) = [str2double(grandData{i}.name), grandData{i}.trialinfo(1)];
end

namesTrials_ord = cat(1,NAMES{1}',NAMES{13}');
for i = 1:length(namesTrials_ord)
   indNames(i) = find(namesTrials_ord(i)==nameTrials(:,1)); 
end

namesTrials_ord(:,2) = nameTrials(indNames,2); %have the right order of first trials 
%%
% for rebaselining/reref the data
% DATA_baselined = baselineData(DATA,round(baselineWin*256+1));
% DATA_rerefed = rerefData(DATA_baselined, refChan);


prestim = PARAMS.prestim;
PARAMS.prestim = 0.2;
CHAN = {'C1','C2','Cz'}; 
C =  [12, 49, 48];
%refChan = find(strcmp(ERPb{1}{1}.label,PARAMS.refchannel));
P1vis = round([0.09 0.11]*256+PARAMS.prestim*256); %lims for max
N1vis = round([0.18 0.22]*256+PARAMS.prestim*256);
N1aud = round([0.05 0.15]*256+PARAMS.prestim*256);
P2aud = round([0.1 0.25]*256+PARAMS.prestim*256);
%%
baselineWin=[0.2 0.25];
lims = P2aud; %change according to desired component
% For the first trials, regardless of condition
[PVAL1,RHO1,smoothedSig] = entrainHabit(DATA_rerefed,namesTrials_ord,condition,C,lims,1); %extreme = max=1, min=2
% For trials in a specific condition
condition = 1;
[PVAL1,RHO1,smoothedSig] = entrainHabitCond(DATA_rerefed,condition,C,lims,1,'all'); %extreme = max=1, min=2

% for i = 1:2
%     for j = 1:length(smoothedSig{i})
%         averagedHabit{i}(j,:) = smoothedSig{i}{j}(1:43);
%     end
% end

% fisher-zscored
RHO1(2,end-1) = NaN;
fZ = [atanh(RHO1(1,:));atanh(RHO1(2,:))];
nanmean(fZ,2)
permutationTest(RHO1(1,:),RHO1(2,:),100000)
permutationTest(fZ(1,:),fZ(2,:),100000)

figure; 
%st_boxdotplot([1:2],{RHO1(1,:);RHO1(2,1:end-1)},[0,0,0;0.5,0.5,0.5],'iqr',[],[],[],[],25,[],[],1)
st_boxdotplot([1:2],{fZ(1,:);fZ(2,1:end-1)},[0,0,0;0.2,0.8,0.5],'iqr',[],[],[],[],55,[],[],1)

%%
legend('TD','ASD')
xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt-0.2)




%% run permutationTestCluster.m below: change number of subjects according to the ERP below 
%%%%%%%%%%%%%%%
Cond1 = 12;
Cond2 = 24;

[pos,neg] = permutationTestCluster(ERPb{Cond1},ERPb{Cond2},PARAMS);
%
sigVals = zeros(size(neg,1),size(neg,2));
% negative/positive clusters are the significant ones; 
sigVals(neg)=1;
sigVals(pos)=1;
sigVals = logical(sigVals);

% SCP (t-test)
% change existing order of channel to an anatomical one
chanLabels = channelsArea;
oldChanOrder = ERPb{1}{1}.label;
newChanOrder = chanLabels; 
for i = 1:64%length(oldChanOrder)
    newChansInds(i) = find(strcmp(ERPb{1}{1}.label, newChanOrder{i}));
end

oldData = {ERPb{Cond1},ERPb{Cond2}};
data = oldData;
FIG = figure;
subplot(4,4,[1 12]);

    a = length(data)/2;  % number of figures;
    %figure; 
for i = 1:length(data)/2
    group1 = data(i);
    group2 = data(i+length(data)/2);
    %MAT = cat(2,ERPb(group1),ERPb(group2));
    MAT = cat(2,group1,group2);
    TIME = [-PARAMS.prestim:0.1:PARAMS.poststim]; %epoch
    RESHAPE = transDataMatricesGroup(MAT); %reshape the data to make t test permutation stat
    tVal = SCP(FIG, RESHAPE,newChansInds, i, TIME,a,sigVals);
    ttestLabels = ERPb{1}{1}.label(1:4:64);
    %set(gca,'Ytick',[1:4:64],'yticklabel',ttestLabels)
end
set(gca,'fontsize', 14);
title(['T-test map for VEP comparison',titles(Cond1)])
box off;
% plot the difference between two conditions
%con = 1;
sig1 = average{Cond1};
sig2 = average{Cond2}; 
Diff = mean(sig1,1) - mean(sig2,1);
diffFig = randn(1);
figure(FIG); 
subplot(4,4,[13, 16]);
plot(x{Cond1},Diff, 'k', 'LineWidth',2); 
ylim([-5 5])
hold on; 

sigValsSum = sum(sigVals);    %find significant values in the difference signal
logSigValSum = logical(sigValsSum);
plot(x{i}(find(logSigValSum)), Diff(find(logSigValSum)), 'r','LineWidth',2);
box off

%% TOPOPLOT change scale of the x-axis in waveforms 
figure(5)
ax = gca;
ax.XAxis.TickLabels = [-PARAMS.prestim:0.2:PARAMS.poststim];
title('TD  - VEP');
colormap('jet')
caxis([-1.5 3]);
set(gca,'fontsize', 14);

% change scale of y-axis to be the same
%ax.YLim = 
figure(6)
ylim(ax.YLim);
title('ASD  - VEP');
colormap('jet')
caxis([0 5]);
set(gca,'fontsize', 14);


%% Present FFT and Low-pass (to show entrainment to rhythm)
LP = 1.9;
SB = 4;
lpFiltData = lpFiltSignals(x,newERPb,C,FS,LP,SB,R,Signal);

%% high pass (without fieldtrip)
fig1 = figure;
for i = 1:length(grandAvg)
    if i < 8
        %fig1
        subplot(length(grandAvg)/2,1,condOrd(i));
        plot(x{i},mean(grandAvg{i}.avg(C,:)),'k');
        ylim([-10 10])
        xlim([0 5.3])
        hold on; 
        for j = 1:length(R{condOrd(i)})
           line([R{condOrd(i)}(j)  R{condOrd(i)}(j)],y,'Color','k');
        end
        hold on; 
    else
        subplot(length(grandAvg)/2,1,condOrd(i));
        plot(x{i},mean(grandAvg{i}.avg(C,:)),'Color',colors(i,:));
        hold on; 
        for j = 1:length(R{condOrd(i)})
           line([R{condOrd(i)}(j)  R{condOrd(i)}(j)],y,'Color','k');
        end
        hold on; 
    end
        
end


%% spectrograms and ITPC

cond = [1,13];
CHAN = 'Oz';
freqoi = 1.5:30;
lines = 0.2; %R{cond(1)};
itcGroup = EntrainmentItpc(DATA,ERPb,cond, CHAN,freqoi,FS,PARAMS, lines); % calc ITPC

tfrGroup = EntrainmentTFR(DATA,ERPb,cond, CHAN,freqoi,FS,PARAMS, lines);% calc TFR

%% Behavior
% to plot from RT Visual data strcture
% reactionTime_bdf_entrainment (in GENERAL ANALYSIS FILES) was used to
% generate Behavior.mat Structure.
load('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis/Behavior/hitRt.mat')
addpath('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis/Behavior');
plotRT(hitRt,0);  %mod: 1:vis 2: aud 3: don't write and save csv

load('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Data Structures/Behavior.mat');

% TD_id: 1 for TD, 2 for ASD
BS = num2str(Behavior.subjects);
for i = 1:length(BS)
    S = num2str(Behavior.subjects(i));
    if strcmp(S(1:2),'10') || strcmp(S(1:2),'12')
        ID(i,1) = 1;
    else
        ID(i,1) = 2;
    end
end

Accmean_TD = Behavior.accuracy(ID==1);
Accmean_ASD = Behavior.accuracy(ID==2);

[mean(cell2mat(Accmean_TD)) mean(cell2mat(Accmean_ASD))]
[std(cell2mat(Accmean_TD)) std(cell2mat(Accmean_ASD))]

[p,h,stats] = ranksum(cell2mat(Accmean_TD),cell2mat(Accmean_ASD))

%false alarms
FA_TD = Behavior.fa(ID==1);
FA_ASD = Behavior.fa(ID==2);
% meanFA_TD = mean(cell2mat(FA_TD'))
% meanFA_TD(2) = sum(meanFA_TD(2:3)); %make 2 and 3 condition#3(interchanging) and 4-8 cond 3 (gradual). Then delete the empty
% meanFA_TD(4) = sum(meanFA_TD(4:8));
% meanFA_TD([3,5:8])=[];
% 
% meanFA_ASD = mean(cell2mat(FA_ASD'))
% meanFA_ASD(2) = sum(meanFA_ASD(2:3)); %make 2 and 3 condition#3(interchanging) and 4-8 cond 3 (gradual). Then delete the empty
% meanFA_ASD(4) = sum(meanFA_ASD(4:8));
% meanFA_ASD([3,5:8])=[];

allSubFA_TD = cell2mat(FA_TD'); 
allSubFA_TD = mean(allSubFA_TD(:,[1,9,11,12]),2); %only on the 4 conditions in the paper
allSubFA_ASD = cell2mat(FA_ASD');
allSubFA_ASD = mean(allSubFA_ASD(:,[1,9,11,12]),2);
%
[mean(allSubFA_TD) mean(allSubFA_ASD)] 
[std(allSubFA_TD)./sqrt(19) std(allSubFA_ASD)./sqrt(19)]
[p,h,stats] = ranksum(allSubFA_TD,allSubFA_ASD)

%misses
miss_TD = Behavior.miss(ID==1);
miss_ASD = Behavior.miss(ID==2);
% meanMiss_TD = mean(cell2mat(miss_TD'))
% meanMiss_TD(2) = sum(meanMiss_TD(2:3)); %make 2 and 3 condition#3(interchanging) and 4-8 cond 3 (gradual). Then delete the empty
% meanMiss_TD(4) = sum(meanMiss_TD(4:8));
% meanMiss_TD([3,5:8])=[];

% meanMiss_ASD = mean(cell2mat(miss_ASD'))
% meanMiss_ASD(2) = sum(meanMiss_ASD(2:3)); %make 2 and 3 condition#3(interchanging) and 4-8 cond 3 (gradual). Then delete the empty
% meanMiss_ASD(4) = sum(meanMiss_ASD(4:8));
% meanMiss_ASD([3,5:8])=[];
allSubMiss_TD = cell2mat(miss_TD'); 
allSubMiss_TD = mean(allSubMiss_TD(:,[1,9,11,12]),2); %only on the 4 conditions in the paper
allSubMiss_ASD = cell2mat(miss_ASD');
allSubMiss_ASD = mean(allSubMiss_ASD(:,[1,9,11,12]),2);

[mean(allSubMiss_TD) mean(allSubMiss_ASD)] %only on the 4 conditions in the paper
[std(allSubMiss_TD)./sqrt(19) std(allSubMiss_ASD)./sqrt(19)]
[p,h,stats] = ranksum(allSubMiss_TD,allSubMiss_ASD)

%number of hits (calculated from Behavior.RT from each trial individually)
hitMean_Sub = cellfun(@size,Behavior.RT,'UniformOutput',false)
hitMean_Sub_TD = hitMean_Sub(ID==1,:);
hitMean_Sub_ASD = hitMean_Sub(ID==2,:);

%mean number of hit, calculated on number of reaction time entries
hitMean_Sub_TD_mean = mean(cell2mat(hitMean_Sub_TD(:,[1,9,11,12])));
hitMean_Sub_ASD_mean = mean(cell2mat(hitMean_Sub_ASD(:,[1,9,11,12])));

hitMean_Sub_TD_sem = std(cell2mat(hitMean_Sub_TD(:,[1,9,11,12])))./sqrt(19);
hitMean_Sub_ASD_sem = std(cell2mat(hitMean_Sub_ASD(:,[1,9,11,12])))./sqrt(19);

meanHit = [mean(hitMean_Sub_TD_mean([1,3,5,7]))  mean(hitMean_Sub_ASD_mean([1,3,5,7]))] %the 1 3 5 7 is to ignore the trigger codes
semHit = [mean(hitMean_Sub_TD_sem([1,3,5,7]))  mean(hitMean_Sub_ASD_sem([1,3,5,7]))]

allSubHit_TD = cell2mat(hitMean_Sub_TD(:,[1,9,11,12]));
allSubHit_ASD = cell2mat(hitMean_Sub_ASD(:,[1,9,11,12]));
[p,h,stats] = ranksum(mean(allSubHit_TD(:,[1,3,5,7]),2),mean(allSubHit_ASD(:,[1,3,5,7]),2))

%max hit number, calculated on number of reaction time entries
hitMean_Sub_TD_max = max(cell2mat(hitMean_Sub_TD(:,[1,9,11,12])));
hitMean_Sub_ASD_max = max(cell2mat(hitMean_Sub_ASD(:,[1,9,11,12])));
maxHit = [max(hitMean_Sub_TD_max([1,3,5,7]))  max(hitMean_Sub_ASD_max([1,3,5,7]))]
semMaxHit = [std(hitMean_Sub_TD_max([1,3,5,7]))./sqrt(19)  std(hitMean_Sub_ASD_max([1,3,5,7]))./sqrt(19)]
[p,h,stats] = ranksum(max(allSubHit_TD(:,[1,3,5,7]),[],2),max(allSubHit_ASD(:,[1,3,5,7]),[],2) )

%min hit number, calculated on number of reaction time entries
hitMean_Sub_TD_min = min(cell2mat(hitMean_Sub_TD(:,[1,9,11,12])));
hitMean_Sub_ASD_min = min(cell2mat(hitMean_Sub_ASD(:,[1,9,11,12])));
minHit = [min(hitMean_Sub_TD_min([1,3,5,7]))  min(hitMean_Sub_ASD_min([1,3,5,7]))]
semMinHit = [std(hitMean_Sub_TD_min([1,3,5,7]))./sqrt(19)  std(hitMean_Sub_ASD_min([1,3,5,7]))./sqrt(19)]
[p,h,stats] = ranksum(mean(allSubHit_TD(:,[1,3,5,7]),2),mean(allSubHit_ASD(:,[1,3,5,7]),2))

%% to run the script to analyze the RT from bdf files
params_b = params_behavior_Entrainment;
if location == 2 && sense == 1
    rawDataPath = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Visual';
elseif location == 2 && sense == 2
    rawDataPath = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Auditory';
    %'/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Visual';
elseif location == 1 && sense == 1
    rawDataPath = 'C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\entrainmentRawData\Visual';
end
    [distributionRange, normMeanRT, anticipInd, anticipators,subjects,RT,meanRT,meanAll,A, STDmat] = reactionTime_bdf_entrainment(location,sense,params_b,rawDataPath);    
% this function saves Behavior structure. Don't run unless it's not in the
% folder

%% Frequency analysis - FIX
cfg = [];
cfg.method =  'wavelet';
cfg.output = 'pow';
cfg.channel = {'O1','O2','Oz'};
%[freq] = ft_freqanalysis(cfg)


%%  fft - plots idividual subjects fft - doesn't work for some cases. See section below it. 
% pwelch spectral analysis 
PLOT = 1;
win = [round(0.5*FS),length(DATA{1}{1}{1})]; %from 0 to the end to the trial
DATAselect = [DATA(1),DATA(4),DATA(6),DATA(7),DATA(8),DATA(11),DATA(13),DATA(14)];
clear fft_mean
for j = 1:length(DATAselect)
    DATA_cond = DATAselect{j};
    [f, entrainSpec] = spectralAnal(DATA_cond,win,0,FS,C(2),2,1,[]); %refers to: Data, data avg, FS, electrodes, casenum(1-3),plotting (yes-1, no-0).
    %maxPower{j} = max(entrainSpec(:,120:160),[],2); %max at 1.5hz (located at samples 100:200);
    if PLOT
        figure;
        plot(f,entrainSpec'); hold on;
        hold on;
        plot(f,nanmean(entrainSpec),'k','LineWidth',2);
        title(num2str(j))
        xlim([0.7 13])
        ylim([0 4]);
    end
    fft_mean(j,:) = nanmean(entrainSpec);
end

%% fft on ERPb
figure;

%
%DAT = newERPb{1};
DATreref = rerefData(DATAselect,37);
DAT = DATreref;
%[spectrum,f] = pwelch(mean(data{i}{j}(elec,win),1),700,600,0.1:0.01:20,Fs); % pwelch(mean(data{i}{j}(elec,:),1),700,600,0:0.1:7,Fs);
clear spectrum
for i = 1:length(DAT)    %sub
    for j = 1:length(DAT{i})            %subjects
        for k = 1:length(DAT{i}{j})    %trials
             [spectrum{i}{j}(k,:),f] = pwelch(mean(DAT{i}{j}{j}(C,:),1),1200,1100,0.1:0.01:20,FS); % pwelch(mean(data{i}{j}(elec,:),1),700,600,0:0.1:7,Fs)
        end
    end
    meanSpectrum{i} = mean(cell2mat(spectrum{i}'));
end
groupSpect = meanSpectrum';

hold on; plot(f,mean(cell2mat(groupSpect)));

%% stats on fft peaks
%power analysis based on means and std of peaks at 1.5 hz for jitter small
%(maxPower{6},maxPower{13})

powerAnalysis('t', 3.56, 2.11,2.49,0.8);


% [f,entrainSpec] = spectralAnal(data,dataAvg,Fs,elec,casenum,plotting,colors);
%%
opol=3;
for i = 1:size(fft_mean,1)
    dtfft_mean(i,:) = detrendNL(f,fft_mean(i,:),opol,0);
    
end

figure; 
conds=[1,4,6,7];
titles = {'Isochronous','Random','Jitter S','Jitter L'};
condsNum = length(conds);
for i = 1:condsNum
    subplot(condsNum,1,i)
    %plot(f, dtfft_mean(conds(i),:),'Color',colors(conds(i),:),'LineWidth',2);
    semilogx(f,10*log10(fft_mean(conds(i),:)),'Color',colors(conds(i),:),'LineWidth',2);
    hold on; 
    %plot(f, dtfft_mean(conds(i)+7,:),'k');
    semilogx(f, 10*log10(fft_mean(conds(i)+7,:)),'k');
    title(titles{i})
    xlim([0.7 15])
    ylim([0 15])

   % New_XTickLabel = {'10^0','10^1'};
     New_XTickLabel = [1,10];
% 
     set(gca,'XTickLabel',New_XTickLabel);
    %ylim([-10 8]);
    legend('TD','ASD')
   set(gca, 'fontsize',14)
end
%New_XTickLabel = get(gca,'xtick');


%% Prepare data for phase locking analysis
