% This code is the main pipline for processing and cleaning of the EEG data
% for Entrainment project. Shlomit Beker 2018-2020 

% settings to select directory of data (comment out if unnecessary)

    clear; clc;
    
    applyICA = 0;     % change to 1 if ICA is needed
    prompt1 = 'At Work? (1-Y / 2-N)? ';
    location = input(prompt1);
    prompt2 = 'Visual (1) / Auditory (2)? ';
    sens = input(prompt2);
    prompt3 = 'Low Pass (1) / Entrainment-WB (2) / ERP (3) /Resting State (4)/ CNV (5) / CNV unique(6) / long ERP (7) or RT (8)? ';
    Filter = input(prompt3);
    prompt4 = 'review bad channels?  1-Yes, 0 - Auto ';
    Review = input(prompt4); 
    prompt5 = 'visually review all trials ? 1-Yes, 0 - no ';
    visInspect = input(prompt5); 

% change the directories 

    if location == 1
        addpath('C:\Users\sbeker\Dropbox (EinsteinMed)\GENERAL ANALYSIS FILES');
        MainFolder = 'C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Processed';
        startup;
        cd('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Analysis');


    else
        addpath('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
        MainFolder =  '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Processed';
        startup;
        cd('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis');
    end

    if sens == 1 && Filter == 3 
        PARAMS = paramsERPVis;
        subFolder = 'EntrainProcessed_ERP vis';
    elseif sens == 1 && Filter == 7 
        PARAMS = paramsERPVisLong;
        subFolder = 'EntrainProcessed_ERPLong vis';
    elseif sens == 2 && Filter == 3 
        PARAMS = paramsERPAud;
        subFolder = 'ERP aud';
    elseif sens == 1 && Filter == 2
        PARAMS = paramsEntrainmentVis;
        subFolder = 'ERP long vis';
    elseif sens == 2 && Filter == 2
        PARAMS = paramsEntrainmentAud;
        subFolder = 'ERP long aud';
    elseif Filter == 4
        PARAMS = paramsRestState;
        subFolder = 'Resting state';
     elseif sens == 1 && Filter == 5
         PARAMS = paramsCNV_Vis;
         subFolder = 'CNV Vis';
     elseif sens == 2 && Filter == 5
         PARAMS = paramsCNV_Aud;
         subFolder = 'CNV Aud';
     elseif sens == 1 && Filter == 6
         PARAMS = paramsCNV_Vis_unique;
         subFolder = 'CNV_Vis_unique';
     elseif sens == 1 && Filter == 8
         PARAMS = paramsERPVisLong_RT;
         subFolder = 'ERP_long_RT';
    end
    
    func =PARAMS.trialfun;

% Make EEG layout

    sfpfile = 'BioSemi64.sfp';
    layout = PARAMS.layout;
    [layout1,elec] = createLayout(sfpfile,1,layout);              % Creating layout and discarding the externals. last var - plot(1) or not(0).

%%% ought to solve a weird fieltrip bug with channel names
    elec.label{1} = 'Fp1';
    elec.label{33} = 'Fpz';
    elec.label{34} = 'Fp2';

    if sens == 1 && location == 2
        allDataPath = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Visual');
    elseif location == 2 && sens == 2
        allDataPath = ('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Auditory');
    elseif sens == 1 && location == 1
        allDataPath =('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\entrainmentRawData\Visual');
        addpath('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Analysis');
    elseif sens == 2 && location == 1    
        allDataPath =('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\entrainmentRawData\Auditory');
        addpath('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Analysis');
    end
    
addPath;    
    
    cd(allDataPath);
    subjectFile = dir(allDataPath);

%% Loop on subjects

    for k =  4:size(subjectFile,1)-1                      % start with the first subject in subjectFile
        trialinfo{k} = [];
        clear event value blockData segmentedRawData
        currentSub = subjectFile(k).name;
        currentPath = [allDataPath,'/',currentSub];
        cd(currentPath);
        bdfFiles = dir('*.bdf');
        numfiles = length(bdfFiles);
        if isempty(numfiles) ~=0
            continue
        end

        VALUE = [];
        SAMPLE = [];

% Loop on bdf files for every participant 
        
    for i =1:numfiles
            i
            event = ft_read_event(bdfFiles(i).name); %all events in the bdf file
            if [event(1).value] > 6000
                A = [event.value];
                A = A-63488;
                B = [0,0];
                A1 = A(1); A2 =  A(2:end);
                Aall = cat(2,A1,B,A2);
                for q = 1:length([Aall])
                    if strcmp('STATUS', {event(q).type})
                        event(q).value = Aall(q);
                    end
                end
            end
            value  = [event(strcmp('STATUS', {event.type})).value]';
            value = fix42(value);
            sample  = [event(strcmp('STATUS', {event.type})).sample]';

            if ~ismember(PARAMS.eventvalue, value)
                    continue
            end
            VALUE = cat(1,VALUE,value);
            SAMPLE = cat(1,SAMPLE,sample);

% Preprocessing

            cfg = [];                                   % Set configurtion variable cfg
            cfg.dataset = [bdfFiles(i).name];
            cfg.continuous = 'yes';
            cfg.lpfilter = 'yes';
            cfg.hpfilter = PARAMS.hpfilter;
            cfg.lpfreq = PARAMS.lpfreq;
            cfg.hpfreq = PARAMS.hpfreq;
            cfg.hpfiltord = PARAMS.hpfiltord;
            cfg.baselinewindow = PARAMS.baselinewindow;
            cfg.demean = PARAMS.demean;
            cfg.detrend =  PARAMS.detrend;
            cfg.reref = PARAMS.reref;
            if strcmp(cfg.reref,'yes')
                cfg.refchannel = PARAMS.refchannel;
            end
            filtContData = ft_preprocessing(cfg);                 % filtered data (on continuous data)

            cfg = [];
            cfg.params = PARAMS;
            cfg.dataset = [bdfFiles(i).name];
            cfg.trialdef.eventvalue = PARAMS.eventvalue;
            cfg.trialdef.eventtype = PARAMS.eventtype;
            cfg.trialdef.prestim = PARAMS.prestim;     % (seconds)          
            cfg.trialdef.poststim = PARAMS.poststim;
            name = bdfFiles(i).name;
            cfg.trialfun = PARAMS.trialfun;
            trl = ft_definetrial(cfg);                                   % epoched data
            if Filter == 2
                samp = sampfinder(cfg);  %find exact timing within the long segments
                trl.samp = samp;
            end
            cfg = [];
            cfg.trl = trl.trl; 
            
            segmentedRawData = ft_redefinetrial(cfg,filtContData)
            trialinfo{k} = cat(1,trialinfo{k}, trl.trl(:,3:4));         
            cfg = [];
            cfg.resamplefs = PARAMS.resample;
            blockData{i} = ft_resampledata(cfg, segmentedRawData);
            if exist('samp')
                blockData{i}.samp = samp;
            end
    end
        
    if isempty(VALUE)==1
        disp('No trigger codes');
        continue
    end
        %segmentedRawData.trialinfo = trialinfo;
        y = filtContData.trial{1}(47,:)';
        x = [1:length(y)]./512;
        figure; plot(x,y,'LineWidth',1.5); ylabel('volt (microv)'); xlabel('Sec'); set(gca,'fontsize', 12);

        empty = [];
        for i = 1:length(blockData)
            if isempty(blockData{i})
                empty = cat(1,empty,i);
            end
        end
        blockData(empty) = [];

        % save all data blocks in a big structure. Next, append all data together for the cleaning and ICA.
        if exist('blockData') == 0
            continue
        end
        l = length(blockData);
        concatData = ft_appenddata([],blockData{1:l});

        %% visual inspection & manual cleaning (cont.)

        if visInspect == 1
                cfg = [];
                cfg.layout = PARAMS.layout;
                cfg.detrend = 'yes';
                cfg.viewmode = 'vertical';
                cfg.channel = 1:28;
                cfg.yaxis = PARAMS.yaxis;
                cfg = ft_databrowser(cfg,concatData);

                concatData = ft_rejectartifact(cfg,concatData);
                totalTrialsAccepted = length(concatData.trial);
                totalTrialRejected = length(badTrials) + length(goodTrials)- totalTrialsAccepted;
        end

        %% Plot to see bad channels
        cfg = [];
        cfg.vartrllength = 1;
        timelock = ft_timelockanalysis(cfg, concatData);

        cfg = [];
        %cfg.xlim = [-PARAMS.prestim PARAMS.poststim];
        %cfg.ylim = [-1e-13 3e-13];
        cfg.layout = PARAMS.layout;

        badChan = findBadChans(timelock.avg(1:64,:)',[],2,2);

        if strcmp(PARAMS.reref,'yes') == 1
            for z = length(PARAMS.refchannel)
                BC = find(strcmp(PARAMS.refchannel(z),concatData.label));
                badChan(find(badChan == BC)) = []; %delete the ref channel from the list of bad channels (always there since it's a flat line)
        
            end
        end

        if Review == 1
            badChan
            figure; ft_multiplotER(cfg, timelock);
            prompt = 'Add bad channels? [] if none  ';
            addBadChan1 = input(prompt);      %add numbers
            badChan = cat(2,badChan,addBadChan1);
        end

        addBadChan2 = 0;

         %% Artifact rejection
        while addBadChan2 == 0;
            badChannels = cell(1,length(badChan));
            for c = 1:length(badChan)
                badChannels(c) = timelock.label(badChan(c));
            end
            
            CleanData = concatData; %NEW save concatData as a new structure CleanData, on which to do the interpolation 
            % interpolation for bad channels
            cfg = [];
            cfg.method = 'triangulation';
            cfg.layout = PARAMS.layout;
            neighbours = ft_prepare_neighbours(cfg, concatData);

            % repair the bad channels with neighbors
            cfg = [];
            cfg.neighbours = neighbours;
            cfg.elec = elec;
            cfg.badchannel = badChannels;
            cfg.layout = PARAMS.layout;

            dataInterp = ft_channelrepair(cfg, concatData); %dataInterp contain all the data
            if Review == 1
                badChan
                figure; ft_multiplotER(cfg, dataInterp);

                prompt6 = 'Add more channels? ([]-no, otherwise - type chan number) ';
                addBadChan2 = input(prompt6);
                badChan = cat(2,badChan, addBadChan2);
                if isempty(addBadChan2) == 1   %NEW
                    addBadChan2 = 1;
                else
                    addBadChan2 = 0;
                end
            else
                addBadChan2 = 1;
            end

        end
        badChannels = cell(1,length(badChan));
        for c = 1:length(badChan)
            badChannels(c) = timelock.label(badChan(c));
        end
        cfg.badchannel = badChannels;
        dataInterp = ft_channelrepair(cfg, CleanData); %Channel repair is now done on the Clean data,
        %%
        %concatData = dataInterp;
        badTrials = [];
        for ii = 1:length(dataInterp.trial)
            c = max(abs(dataInterp.trial{ii}(1:PARAMS.numelec,:)),[],2); % for each trial, calc the max(abs)

            d = length(find(c>PARAMS.thres));
            if d > 30                                                % find minimum elec that cross the threshold
                badTrials = cat(1, badTrials, ii);                   % disselect this trial.
            end
        end

        goodTrials = 1:length(dataInterp.trial);
        goodTrials(badTrials) = [];
        cfg = [];
        cfg.trials = goodTrials;
        %cfg.trials = 'all';
        cfg.channel = 1:PARAMS.numelec;
        dummy = ft_preprocessing(cfg, dataInterp);
        trialinfo{k}(badTrials,:) = [];
        dummy.trialinfo = trialinfo{k};
        % look for rows of nans (=bad channels that were not replaced) and
        % interpolate
        
        for m = 1:length(dummy.trial)
            [a,b] = find(isnan(dummy.trial{m}));
            a = unique(a);
            if isempty(a)==0
                if ~ismember(64,a) && ~ismember(1,a)
                    dummy.trial{m}(a,:) = mean([dummy.trial{m}(a-1,:);dummy.trial{m}(a+1,:)],1)
                end
            end
        end
        
        % get rid of nans in the trials
        nanTrials = nanSearch(dummy);
        if isempty(nanTrials) ==0
            goodTrials(nanTrials) = [];
            cfg = [];
            cfg.trials = goodTrials;
            cfg.channel = 1:PARAMS.numelec;
            dummy = ft_preprocessing(cfg, dataInterp);
            trialinfo{k}(nanTrials,:) = [];
            dummy.trialinfo = trialinfo{k};

        end

       
      %% ICA                downsampling the data, choosing a subset of trials.
        if applyICA == 1
           ICAdata = ICA(PARAMS, dummy); 
        end
        
        %%
        %cd([MainFolder,'/',subFolder,'/',num2str(sens)]);
        cd([MainFolder,'/',subFolder]);

        if applyICA == 1
              save(['Procdata_',currentSub], 'ICAdata');
        else   
              save(['Procdata_',currentSub], 'dummy');
        end
    end
    
