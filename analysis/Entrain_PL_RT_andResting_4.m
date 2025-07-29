
 
% Phase Locking (Intra-Trial-Phase-Coherence) analysis and interaction with
% RT. Shlomit Beker 2023
% Run Main_Analysis_Entrainment with Phase-RT (8).
% runs on selected channels. 

%% parameters for phase locking
%startup;

clear PLallstim PLV Phase

CHAN = {'FC1','FCz','FC2'}; % for the collapsed-across-participant figure in the paper (colorful). 
%CHAN = {'O1','Oz','O2'}; %for RT-Phase analysis

%CHAN = {'C1','Cz','C2'}; % for entrainment: 'C1','Cz','C2' for visual sequence: 'AF3','Fp1','AFz','Fpz','A7','PO3','POz','PO4',

CHANNELS = [find(strcmp(ERP{1}{1}.label,CHAN{1})),find(strcmp(ERP{1}{1}.label,CHAN{2})),...
    find(strcmp(ERP{1}{1}.label,CHAN{3}))];

SAMP_RATE = 256; 
LOW_FREQUENCY = 0.6;
HIGH_FREQUENCY = 43; %range of frequencies on which to make the coherence
OMEGA = 6;
LENGTH_WIN = length(DATA{1}{1}{1})./SAMP_RATE;   
TIME_WINDOW = 1:LENGTH_WIN*SAMP_RATE;
COLORS = [inferno(7);viridis(7)];
clear i;
FOI = 17:20; %location of 1.5Hz in the frequencies vector
%FOI = 40:46; %location of Theta (4-7Hz) in the frequencies vector
%FOI = 52:55; %location of Alpha (9-11Hz) in the frequencies vector

start = 768; % stim onset 3 sec. (RT/Phase epochs are [-3:3]).
STIM_TIMES = start;
layout = PARAMS.layout;


%% reref and baseline DATA
if Signal == 6 || Signal == 8
    DATAa = DATA([1,13,11,23,12,24,9,21]);
    %DATAa = [DATA(1),DATA(13),DATA(11),DATA(23),DATA(12),DATA(24),DATA(9),DATA(21)];
    names = NAMES([1,13,11,23,12,24,9,21]);
    RTsel = RT([1,13,11,23,12,24,9,21]);
elseif Signal == 7
    DATAa = DATA([1,4,6,7,9,13,15,16]);
    names = NAMES([1,4,6,7,9,13,15,16]);
end

%% For long resting state - 30s

DATAa  = DATA([17,19]); %for the long resting states (from the 30 seconds
%at the beginning of the experiment)
names = NAMES([17,19]);
LENGTH_WIN = length(DATA{17}{1}{1})./SAMP_RATE;   

%%

%DATAr = rerefData(DATAa,'all');   % rerefed DATA is used in the paper (April 2024).
DATAr = DATAa; 
LENGTH_WIN = length(DATA{1}{1}{1})./SAMP_RATE;   
TIME_WINDOW = 1:LENGTH_WIN*SAMP_RATE;


%limit number of trials - for Signals 6,8
if Signal == 6 || Signal == 8
    DATArr = DATAr
    for i = 1:length(DATArr)
        for j = 1:length(DATArr{i})
            if length(DATArr{i}{j})<20
                DATArr{i}{j} = [];
            end
        end
        emptycells = find(cellfun(@isempty,DATArr{i}));
        DATArr{i}(emptycells) = [];
        RTsel{i}(emptycells) = [];
    end
    DATAr = DATArr;
end

% a table with number of trial for each participant
for m = 1:length(DATAr)
    for n = 1:length(DATAr{m}) 
        trialNum(m,n) = length(DATAr{m}{n}); 

    end
end


%% With Specific Channels. Output:  data mat files, by condition
CHAN             %spit current Channels 
flag = 0;
clear sumAngles STphase1 Angles Phase PL PLV PLallstim Ntrial
    for COND = 1:length(DATAr)
        for stim = 1:length(STIM_TIMES)
            for participant = 1:length(DATAr{COND})
                clear STphase1
                currentData = DATAr{COND}{participant}; %all trials
                cond = currentData;
                Ntrial = length(cond);
                for i_trial = 1:Ntrial
                    if size(cond{i_trial},2) == size(cond{1},2) %to control shorter trials (ignore them)
                        flag = flag+1;
                        [wave,period,scale,cone_of_influence] = basewave4(squeeze(mean(cond{i_trial}(CHANNELS,:)))',SAMP_RATE,LOW_FREQUENCY,HIGH_FREQUENCY,OMEGA,0);
                        STphase1(:,:,i_trial) = squeeze(angle(wave));
                        %STphase1_degrees(:,:,i_trial) = radtodeg(STphase1(:,:,i_trial));%convert to degrees
                    end
                end
                frequencies = 1./period;
                TOI = ceil(STIM_TIMES(stim));
                 PL = squeeze(mean((abs(sum(exp(1i*STphase1(:,TIME_WINDOW,:)),3))/Ntrial),2));

               % PL = squeeze(mean(abs(sum(exp(1i*STphase1(:,TIME_WINDOW,:),3))/Ntrial),2));
                PLallstim(:) = PL;
                %Phase{stim,COND}{participant} =
                %squeeze(STphase1(FOI,TOI,:)); %for 1 freq point
                Phase{stim,COND}{participant} = mean(squeeze(STphase1(FOI,TOI,:)),1); 

                PLV{COND}{participant} = PLallstim;
            
            end     
        end
    end


    %% WITH ALL CHANNELS (takes 1 hour, upload the existing PLV structure). 
clear Phase STphase1 PLallstim PLV
load(PARAMS.layout);
relevantChannels = ERP{1}{1}.label(1:64);%{'PO3','POz','PO4','Oz','O1','O2','AF3','AFz','AF4','F1','Fz','F2'};
CHANNELS = find(ismember(lay.label,relevantChannels)==1);
Nchan = length(CHANNELS);
tic
flag = 0;
clear sumAngles STphase1 Angles Phase PLallstim PLV
%PLmat = cell(1,10); % the final cell array will include the four conditions (cue*group)
Angles = [];
participantFlag = 0;
for COND = 1:length(DATAr)
    for i_channel = 1:Nchan;
    %for stim = 1:length(STIM_TIMES)
        for participant = 1:length(DATAr{COND})
            clear STphase1
            currentData = DATAr{COND}{participant};
            cond = currentData;
            Ntrial = length(cond);
            for i_trial = 1:Ntrial
                if size(cond{i_trial},2) == size(cond{1},2) %to control shorter trials (ignore them)
                    flag = flag+1;
                    [wave,period,scale,cone_of_influence] = basewave4(squeeze(cond{i_trial}(CHANNELS(i_channel),:))',SAMP_RATE,LOW_FREQUENCY,HIGH_FREQUENCY,OMEGA,0);
                    STphase1{i_channel}(:,:,i_trial) = squeeze(angle(wave));
                    %STphase1_degrees(:,:,i_trial) = radtodeg(STphase1(:,:,i_trial));%convert to degrees
                end
            end
            frequencies = 1./period;
            
            
            % PL = squeeze(mean((abs(sum(exp(1i*STphase1(:,TIME_WINDOW,:)),3))/Ntrial),2));
            % PLallstim(:,i_channel) = PL;

            %%%%%%%%%%%%%%%%%%%%%
%             count = 0;
%             for rand_stim = 1:5
 %                randPL(rand_stim,:) = squeeze(mean((abs(sum(exp(1i*STphase_all_stim(:,TIME_WINDOW,count+[1:Ntrial(rand_stim)])),3))/Ntrial(rand_stim)),2));
%                 count = Ntrial(rand_stim);
%             end
            %%%%%%%%%%%%%%
            Phase{stim,COND}{participant}(:,i_channel) = mean(squeeze(STphase1{i_channel}(FOI,TOI,:)),1);
            
            %%%%%%%%%%%%%%
            

            %PLV{COND}{participant}(:,i_channel)= PLallstim(:,i_channel);
            
        end
        
    end
end


toc

cd('/Users/shlomit/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Data Structures');
cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Data Structures');

%save('Phase_allChannels_forRT_5Hz_refAll.mat','Phase');


% %% z-scoring the RT (don't use for the collapsed across participants as the section below. only for channels/individuals). 
%     RTsel_z = RTsel; 
%     for i = 1:length(RTsel)
%         for j = 1:length(RTsel{i})
%             RTsel_z{i}{j} = zscore(RTsel_z{i}{j}(:,1));
%         end
%     end
%     RTsel = RTsel_z;

    %% Performance (Hit RT) - Phase corr (creates violin plots of the circular correlation) - collapsed across participants - colorful figure (just visualization)
clear bins_means
    figure(100); sgtitle(CHAN)
    centers = [-pi,-3/4*pi,-pi/2, -pi/4,0,pi/4,pi/2,3/4*pi];
    edges = interp1([1,3,5,7,9,11,13,15],centers,[2,4,6,8,10,12,14]);
    edges = cat(2,-pi*5/4,edges, pi)
    Xlabels = {'-pi','-3/4*pi','-pi/2', '-pi/4','0','pi/4','pi/2','3/4*pi'};
    CH_names = ERP{1}{1}.label;

    %edges = (-pi-3/4*pi)/2, (-3/4*pi-pi/2)/2,(-pi/2-pi/4)/2,-pi/4/2,pi/4/2,(-pi/4+pi/2)/2,(-pi/2+3/4*pi)
    clear RT_bins RT_all_cond rho pval
    figOrd = [1,5,2,6,3,7,4,8];
    for k = 1:length(Phase)
        RTmat = cell2mat(RTsel{k}');
        PhaseMat = cell2mat(Phase{k});
        bins = discretize(PhaseMat,edges);
        ord = unique(sort(bins));
        ord = ord(isnan(ord)~=1);
        for j = 1:length(ord)
            RT_bins{k,j} = RTmat(bins==ord(j),1);
        end
        %subplot(2,4,k); 
        figure(100);
        subplot(2,4,figOrd(k))
        [h,L,MX,MED,bw] = violin(RT_bins(k,:));
        %circAmp_cond(k) = max(MX)-min(MX);

        xticklabels(Xlabels)

        bins_means{k} = MX;
        %bins_med = MED;
        %hold on; plot(0:8,[mean(bins_means{k}), bins_means{k},mean(bins_means{k})] ,'k')
        hold on; plot(0:9,[bins_means{k}(1),bins_means{k}, bins_means{k}(end)] ,'k')
        %hold on; plot(0:9,[bins_med(1),bins_med, bins_med(end)],'r')
        [rho(k) pval(k)] = circ_corrcl(centers, MX)
        
        legend off
        text(5,700,[num2str(pval(k)),' ',num2str(rho(k))])
        %ylim([0 900])
        
        text(5,3,[num2str(pval(k)),' ',num2str(rho(k))])
        %ylim([-7 7]) %only if zscores are used
        peakTroughAmp(k) = sineFit(MX,0); %calculate peak-trough amp, and plot(1) or not (0)

    end


    %% Phase-RT collapsed across all 4 conds, per group 

clear RT_bins_TD RT_bins_ASD

for i = 1:8
        RT_bins_TD{1,i} = cell2mat(RT_bins(1:4,i));
        RT_bins_ASD{1,i} = cell2mat(RT_bins(5:8,i));

end
 figure;
% [h,L,MX,MED,bw] = violin(RT_bins_TD);
 [h,L,MX,MED,bw] = violin(RT_bins_TD,'bw',0.15);

circAmp_TD = max(MX)-min(MX);

 xticklabels(Xlabels)
  bins_means = MX;
  hold on; plot(0:9,[bins_means(1),bins_means, bins_means(end)] ,'k')
 [rho pval] = circ_corrcl(centers, MX)

 figure;
 [h,L,MX,MED,bw] = violin(RT_bins_ASD, 'bw', 0.15);
 xticklabels(Xlabels)
 bins_means = MX;
 hold on; plot(0:9,[bins_means(1),bins_means, bins_means(end)] ,'k')
 [rho pval] = circ_corrcl(centers, MX)

circAmp_ASD = max(MX)-min(MX);

 
  %% z-scoring the RT (don't use for the collapsed across participants as the section below.  only for channels/individuals). 
    RTsel_z = RTsel; 
    for i = 1:length(RTsel)
        for j = 1:length(RTsel{i})
            RTsel_z{i}{j} = zscore(RTsel_z{i}{j}(:,1));
        end
    end
    RTsel = RTsel_z;
  
  
  
  %% Performance (Hit RT) - collapsed across participants - per Channel (Use the PLV with ALL channels). 
    %figure; sgtitle('all channels')
    load(PARAMS.layout);
    relevantChannels = ERP{1}{1}.label(1:64);
    CHANNELS = find(ismember(lay.label,relevantChannels)==1);
    Nchan = length(CHANNELS);
    centers = [-pi,-3/4*pi,-pi/2, -pi/4,0,pi/4,pi/2,3/4*pi];
    edges = interp1([1,3,5,7,9,11,13,15],centers,[2,4,6,8,10,12,14]);
    edges = cat(2,-pi*5/4,edges, pi)
    Xlabels = {'-pi','-3/4*pi','-pi/2', '-pi/4','0','pi/4','pi/2','3/4*pi'};
    
    %edges = (-pi-3/4*pi)/2, (-3/4*pi-pi/2)/2,(-pi/2-pi/4)/2,-pi/4/2,pi/4/2,(-pi/4+pi/2)/2,(-pi/2+3/4*pi)
    clear RT_bins RT_all_cond bins rho pval bins_means

    for k = 1:length(Phase)
        figure;
        RTmat = cell2mat(RTsel{k}');
       for m = 1:Nchan
        PhaseMat = cell2mat(Phase{k}');
        PhaseMatChan = PhaseMat(:,m);
        bins{m} = discretize(PhaseMatChan,edges);
        ord = unique(sort(bins{m}));
        ord = ord(isnan(ord)~=1);
        for j = 1:length(ord)
            %RT_bins{k}{j}{m} = RTmat(bins{m}==ord(j),1);
            RT_bins{k}{j} = RTmat(bins{m}==ord(j),1);

        end
        subplot(8,8,m); 
        title(CH_names{m})
        [h,L,MX,MED,bw] = violin(RT_bins{k});
        circAmp_channels{k}(m) = max(MX)-min(MX);
        xticklabels(Xlabels)

        bins_means{k} = MX;
        %bins_med = MED;
        %hold on; plot(0:8,[mean(bins_means{k}), bins_means{k},mean(bins_means{k})] ,'k')
        hold on; plot(0:9,[bins_means{k}(1),bins_means{k}, bins_means{k}(end)] ,'k')
        %hold on; plot(0:9,[bins_med(1),bins_med, bins_med(end)],'r')
        [rho{k}(m) pval{k}(m)] = circ_corrcl(centers, MX)
        %[rho(k) pval(k)] = circ_corrcl(centers, MED)

        legend off
        text(5,700,[num2str(pval{k}(m)),' ',num2str(rho{k}(m))])
        ylim([0 900])
        %ylim([-3 3])
       end
    end

 % plot
 % plotPval_RT_PhaseCorr; % a function to plot the topoplot of pvals % %*******************

 % Stats: rho
Y = cell2mat(rho);

g1 = [ones(1,length(rho{1})),ones(1,length(rho{2}))*2,ones(1,length(rho{3})),ones(1,length(rho{4}))*2,...
    ones(1,length(rho{5})),ones(1,length(rho{6}))*2,ones(1,length(rho{7})),ones(1,length(rho{8}))*2];%group 1-TD 2-ASD
g2 = [ones(1,length(rho{1})),ones(1,length(rho{2})),ones(1,length(rho{3}))*2,ones(1,length(rho{4}))*2,...
    ones(1,length(rho{5}))*3,ones(1,length(rho{6}))*3,ones(1,length(rho{7}))*4,ones(1,length(rho{8}))*4]; % Condition (1-4)

[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

ranksum(rho{4},rho{8}) 
% plot rho (correlation)
Rho_all{1,1} = rho{1};
Rho_all{1,2} = rho{2};
Rho_all{2,1} = rho{3};
Rho_all{2,2} = rho{4};
Rho_all{3,1} = rho{5};
Rho_all{3,2} = rho{6};
Rho_all{4,1} = rho{7};
Rho_all{4,2} = rho{8};

cl = [0,0,0;1,0,0];
figure
h   = rm_raincloud(Rho_all, cl);


% stats and plot circ amp (size of oscillation)

Y = cell2mat(circAmp_channels);

g1 = [ones(1,length(circAmp_channels{1})),ones(1,length(circAmp_channels{2}))*2,ones(1,length(circAmp_channels{3})),ones(1,length(circAmp_channels{4}))*2,...
    ones(1,length(circAmp_channels{5})),ones(1,length(circAmp_channels{6}))*2,ones(1,length(circAmp_channels{7})),ones(1,length(circAmp_channels{8}))*2];%group 1-TD 2-ASD
g2 = [ones(1,length(circAmp_channels{1})),ones(1,length(circAmp_channels{2})),ones(1,length(circAmp_channels{3}))*2,ones(1,length(circAmp_channels{4}))*2,...
    ones(1,length(circAmp_channels{5}))*3,ones(1,length(circAmp_channels{6}))*3,ones(1,length(circAmp_channels{7}))*4,ones(1,length(circAmp_channels{8}))*4]; % Condition (1-4)

[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

circAmpCH{1,1} = circAmp_channels{1};
circAmpCH{1,2} = circAmp_channels{2};
circAmpCH{2,1} = circAmp_channels{3};
circAmpCH{2,2} = circAmp_channels{4};
circAmpCH{3,1} = circAmp_channels{5};
circAmpCH{3,2} = circAmp_channels{6};
circAmpCH{4,1} = circAmp_channels{7};
circAmpCH{4,2} = circAmp_channels{8};

cl = [0,0,0;1,0,0];
figure
h   = rm_raincloud(circAmpCH, cl);


    %% Performance (Hit RT) - Phase corr (creates violin plots of the circular correlation) - individual level
   
    centers = [-pi,-3/4*pi,-pi/2, -pi/4,0,pi/4,pi/2,3/4*pi];
    edges = interp1([1,3,5,7,9,11,13,15],centers,[2,4,6,8,10,12,14]);
    edges = cat(2,-pi*5/4,edges, pi);
    Xlabels = {'-pi','-3/4*pi','-pi/2', '-pi/4','0','pi/4','pi/2','3/4*pi'};
    clear RT_bins RT_all_cond bins_control RT_bins_con circAmp_con_ind MX_all

    for k = 1:length(Phase)   % conditions 
        for m = 3:length(Phase{k})   %participants
            RTmat = RTsel{k}{m}(:,1);
            PhaseMat = Phase{k}{m};
            bins = discretize(PhaseMat,edges);
             % for rr = 1:1000
             %    bins_control(rr,:) = randi(8,[length(PhaseMat),1]);
             % end
            ord = unique(sort(bins));
            eightVec = [1:8];
            miss = ismembertol(eightVec,ord);
            eightVec(miss==0) = nan;
            %ord = ord(isnan(ord)~=1);
            for j = 1:length(eightVec)
                RT_bins{m,k}{j} = RTmat(bins==eightVec(j),1);
                if isempty(RT_bins{m,k}{j}) == 1
                    RT_bins{m,k}{j} = 0; 
                end
                numz{k,m}(j) = nnz(RT_bins{m,k}{j});

            end
            MX_all{k}(m,:) = cellfun(@(x) mean(x(:)), RT_bins{m,k}); %means (instead of plotting violin and calc MX)
            %circAmp_indiv{k}(m) = max(MX)-min(MX);
            %MX_all{k}(find(MX_all{k}==0)) = NaN;
            zloc = find(MX_all{k}(m,:)==0);
            if zloc
                for mm = 1:length(zloc)
                    if zloc(mm)>1 & zloc(mm)<8
                        MX_all{k}(m,zloc(mm)) = mean([MX_all{k}(m,zloc(mm)-1), MX_all{k}(m,zloc(mm)+1)]); % if a bin is empty, make an average of the before and after
                    end
                    if zloc(mm)==1 
                        MX_all{k}(m,zloc(mm)) = MX_all{k}(m,2);
                    end
                    if zloc(mm)==8
                        MX_all{k}(m,zloc(mm)) = MX_all{k}(m,7);
                    end
                end
            end
            [circAmp_indiv{k}(m),R2_composite{k}(m)] = sineFit(MX_all{k}(m,:),0);
            %[rho_ind{k}(m) pval_ind{k,m}] = circ_corrcl(centers, MX);
            clear RT_bins_con bins_control circAmp_con
            for rr = 1:1000
                bins_control(:,rr) = randi(8,[length(PhaseMat),1]);
                for jj = 1:8
                   RT_bins_con{jj} = RTmat(bins_control(:,rr)==jj,1);
                end
                MX_con = cellfun(@(x) mean(x(:)), RT_bins_con);
                [circAmp_con(rr),R2_composite_con(rr)] = sineFit(MX_con,0);
            end
            circAmp_con_ind{k}(m) = mean(circAmp_con);
            R2_comp_con{k}{m} = mean(R2_composite_con);
        end
    end
 
    
%plot violin for each group and cond
figure(200);

for i = 1:8
    subplot(2,4,i)
    [h,L,mx,MED,bw] = violin(nanmean(MX_all{i}));
    %circAmp_cond(k) = max(MX)-min(MX);
    xticklabels(Xlabels)
    bins_means = mx;
    hold on; plot(0:9,[bins_means(1),bins_means, bins_means(end)] ,'k')

end

% to calc proportion of Hit response
    clear numzPercent
for k = 1:size(numz,1)
    for m = 1:size(numz,2)
        numzPercent{k}{m,:} = numz{k,m}./sum(numz{k,m});
        if ~isempty(numzPercent{k}{m,:})
            phaseZero(k,m) = numzPercent{k}{m,:}(5);
        end
    end
    numzPercent_Group{:,k} = mean(cell2mat(numzPercent{k}));
end



   clear numzPercent
for k = 1:size(numz,1)
    for m = 1:size(numz,2)
        numzPercent{k,m} = numz{k,m}./sum(numz{k,m});
        if ~isempty(numzPercent{k,m})
            phaseZero(k,m) = numzPercent{k,m}(5);
        end
    end
end

numzPercent_T = numzPercent';


%% permutation test
mean1 = [];
mean2 = [];
figure
titles = {'TD Iso','ASD Iso','TD SJ','ASD SJ','TD LJ','ASD LJ','TD Rand','ASD Rand'};
for cc = 1:8 % conditions
    for pp = 1:1000
        allData = [circAmp_indiv{cc},circAmp_con_ind{cc}];
        r = randperm(length(allData));
        shuffled = allData(r);
        mean1(pp) = mean(shuffled(1:length(circAmp_indiv{cc})));
        mean2(pp) = mean(shuffled(length(circAmp_indiv{cc})+1:end));
    end
    realdiff = mean(circAmp_indiv{cc})-mean(circAmp_con_ind{cc});
    diff_perm = mean1 - mean2;
    pval_sineFit_permTest(cc) = 1-(length(find(realdiff>=diff_perm))/1000);
    subplot(4,2,cc); histogram(diff_perm); 
    title(titles{cc})
    hold on; line([realdiff realdiff],[0 100],'Color','r'); text(0.3,100,['p= ',num2str(pval_sineFit_permTest(cc))]);

end

%% STATS - Circ Amplitude (distance between max and min of the sine fit to the oscillation) - make sure it's on the z-scored RT. 

AmpSub_cloud{1,1} = circAmp_indiv{1};
AmpSub_cloud{1,2} = circAmp_indiv{2};
AmpSub_cloud{2,1} = circAmp_indiv{3};
AmpSub_cloud{2,2} = circAmp_indiv{4};
AmpSub_cloud{3,1} = circAmp_indiv{5};
AmpSub_cloud{3,2} = circAmp_indiv{6};
AmpSub_cloud{4,1} = circAmp_indiv{7};
AmpSub_cloud{4,2} = circAmp_indiv{8};
%add the shuffled data
% AmpSub_cloud{1,3} = circAmp_con_ind{1};
% AmpSub_cloud{2,3} = circAmp_con_ind{3};
% AmpSub_cloud{3,3} = circAmp_con_ind{5};
% AmpSub_cloud{4,3} = circAmp_con_ind{7};
% AmpSub_cloud{1,4} = circAmp_con_ind{2};
% AmpSub_cloud{2,4} = circAmp_con_ind{4};
% AmpSub_cloud{3,4} = circAmp_con_ind{6};
% AmpSub_cloud{4,4} = circAmp_con_ind{8};


figure;
%cl = rand(2,3);
cl = [0,0,0;1,0,0];

%cl = [0,0,0;1,0,0; 0,1,1;0,0,1];
h = rm_raincloud(AmpSub_cloud, cl);

Y = cell2mat(circAmp_indiv);

g1 = [ones(1,length(circAmp_indiv{1})),ones(1,length(circAmp_indiv{2}))*2,ones(1,length(circAmp_indiv{3})),ones(1,length(circAmp_indiv{4}))*2,...
    ones(1,length(circAmp_indiv{5})),ones(1,length(circAmp_indiv{6}))*2,ones(1,length(circAmp_indiv{7})),ones(1,length(circAmp_indiv{8}))*2];%group 1-TD 2-ASD
g2 = [ones(1,length(circAmp_indiv{1})),ones(1,length(circAmp_indiv{2})),ones(1,length(circAmp_indiv{3}))*2,ones(1,length(circAmp_indiv{4}))*2,...
    ones(1,length(circAmp_indiv{5}))*3,ones(1,length(circAmp_indiv{6}))*3,ones(1,length(circAmp_indiv{7}))*4,ones(1,length(circAmp_indiv{8}))*4]; % Condition (1-4)

[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

%% Control check - same number of trials in each bean
clear RT_bins_collapsed RT_bins_collapsed_select R2_composite_con circAmp_con R2
RT_bins_collapsed = cell(1,8);
for i = 1:size(RT_bins,2)  %conditions
    RT_bins_collapsed{i} = cell(1,8);
    for j = 1:size(RT_bins,1)    %participants
        if ~isempty(RT_bins{j,i})
            for m = 1:8
                RT_bins_collapsed{i}{:,m} = cat(1,RT_bins_collapsed{i}{m},RT_bins{j,i}{:,m});
            end
        end
    end
end
% Take 50 from each bin

for kk = 1:100
    for i= 1:8
        for j = 1:8
            if length(RT_bins_collapsed{i}{j})<50
                RT_bins_collapsed_select{i}{j} = RT_bins_collapsed{i}{j};
            else
                RP = randperm(length(RT_bins_collapsed{i}{j}),50);
                RT_bins_collapsed_select{i}{j} = RT_bins_collapsed{i}{j}(RP);
            end
        end
        %subplot(2,4,figOrd(i))
        [h,L,mx,MED,bw] = violin(RT_bins_collapsed_select{i},[],[],[],[]);
        bins_means = mx;
        %hold on; plot(0:9,[bins_means(1),bins_means, bins_means(end)] ,'k')
        [circAmp_con(i), R2_composite_con(i)] = sineFit(mx,0);
        
        %title(['R² = ', num2str(R2_composite_con(i))])
    end
    R2{kk} = R2_composite_con;
end

R_mean = mean(cell2mat(R2'));


%% stats on number of trials per condition per phase
%2*2
phaseZero(phaseZero==0)=NaN;
Y = cat(2,phaseZero(1,~isnan(phaseZero(1,:))),phaseZero(2,~isnan(phaseZero(2,:))),phaseZero(7, ~isnan(phaseZero(7,:)),:),phaseZero(8, ~isnan(phaseZero(8,:)),:)); 
g1 = [ones(1,length(phaseZero(1,~isnan(phaseZero(1,:))))),ones(1,length(phaseZero(2,~isnan(phaseZero(2,:)))))*2,ones(1,length(phaseZero(7,~isnan(phaseZero(7,:))))),ones(1,length(phaseZero(8,~isnan(phaseZero(8,:)))))*2];%group 1-TD 2-ASD
g2 = [ones(1,length(phaseZero(1,~isnan(phaseZero(1,:))))),ones(1,length(phaseZero(2,~isnan(phaseZero(2,:))))),ones(1,length(phaseZero(7,~isnan(phaseZero(7,:)))))*2,ones(1,length(phaseZero(8,~isnan(phaseZero(8,:)))))*2]; %condition 1-ISOjit 2-jitLRand
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])


%plot
phase_zero_cloud{1,1} = phaseZero(1,~isnan(phaseZero(1,:)));
phase_zero_cloud{1,2} = phaseZero(2,~isnan(phaseZero(2,:)));
phase_zero_cloud{2,1} = phaseZero(7,~isnan(phaseZero(7,:)));
phase_zero_cloud{2,2} = phaseZero(8,~isnan(phaseZero(8,:)));

figure;
%cl = rand(2,3);
cl = [0,0,0;1,0,0];
h = rm_raincloud(phase_zero_cloud, cl);


 
    %% different approach: unite each two conditions (periodic / aperiodic). 
% Deleted from here. Search in prev. versions

%% tuning curve for number of trials (using percentages) - data and plot in the excel ReactionTimePhaseTuningCurve
TD_Iso_JitS = [1.46627566	3.0163385	7.41516548	17.80477587	33.55676581	22.91579388	8.294930876	5.529953917];
TD_JitL_Rand = [2.823408624	6.776180698	10.42094456	17.14579055	21.81724846	19.45585216	11.13963039	10.42094456];
ASD_Iso_JitS = [2.182759896	4.92045875	10.35886053	20.08879023	29.00480947	18.01701813	8.768035516	6.659267481];
ASD_JitL_Rand = [3.700848111	8.404009252	11.79645335	16.8080185	18.19583655	15.61295297	11.52659985	13.95528142];

Y = cat(2,TD_Iso_JitS,TD_JitL_Rand,ASD_Iso_JitS,ASD_JitL_Rand); 
g1 = [ones(1,length(TD_Iso_JitS)),ones(1,length(TD_JitL_Rand)),ones(1,length(ASD_Iso_JitS))*2,ones(1,length(ASD_JitL_Rand))*2];%group 1-TD 2-ASD
g2 = [ones(1,length(TD_Iso_JitS)),ones(1,length(TD_JitL_Rand))*2,ones(1,length(ASD_Iso_JitS)),ones(1,length(ASD_JitL_Rand))*2]; %condition 1-ISOjit 2-jitLRand
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

%% Calc differences in number of trials per group (across conds)
TD_trial_num = [108	388	386	160	201	253	230	317	327	394	209	373	413	232	322	266	320	254 179];
ASD_trial_num = [108	363	260	190	230	239	241	243	311	379	155	377	337	212	289	232	256	231	196];

[p,h,stats] = ranksum(TD_trial_num, ASD_trial_num)


%% Calc reaction times for each condition and participant and plot with raincloud

for ii = 1:length(RTsel)
    for jj = 1:length(RTsel{ii})
        RTSub{ii}(jj) = mean(RTsel{ii}{jj}(:,1));
    end
end

RTSub_cloud{1,1} = RTSub{1};
RTSub_cloud{1,2} = RTSub{2};
RTSub_cloud{2,1} = RTSub{3};
RTSub_cloud{2,2} = RTSub{4};
RTSub_cloud{3,1} = RTSub{5};
RTSub_cloud{3,2} = RTSub{6};
RTSub_cloud{4,1} = RTSub{7};
RTSub_cloud{4,2} = RTSub{8};

figure;
%cl = rand(2,3);
cl = [0,0,0;1,0,0];

h   = rm_raincloud(RTSub_cloud, cl);

Y = cat(2,RTSub{1}, RTSub{2},RTSub{3}, RTSub{4},RTSub{5}, RTSub{6},RTSub{7},RTSub{8}); 
g1 = [ones(1,length(RTSub{1})),ones(1,length(RTSub{2}))*2,ones(1,length(RTSub{3})),ones(1,length(RTSub{4}))*2,ones(1,length(RTSub{5})),ones(1,length(RTSub{6}))*2,...
    ones(1,length(RTSub{7})),ones(1,length(RTSub{8}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(RTSub{1})),ones(1,length(RTSub{2})),ones(1,length(RTSub{3}))*2,ones(1,length(RTSub{4}))*2,...
    ones(1,length(RTSub{5}))*3,ones(1,length(RTSub{6}))*3,ones(1,length(RTSub{7}))*4,ones(1,length(RTSub{8}))*4];%condition 1-ISO 2-jit s 3-jit l
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})

figure; 
results = multcompare(stats,'Dimension',[1 2])

%descriptive

meanGroupCond = cellfun(@mean,RTSub);
semGroupCond = cellfun(@std,RTSub)./sqrt(19);


meanRTAll_TD = mean(meanGroupCond([1,3,5,7]));
meanRTAll_ASD = mean(meanGroupCond([2,4,6,8]));


%% use a different violin plot function

RT_ISO_TD = cell2mat(RT_bins(1,:)');
conds_ISO_TD = strcat(1,Xlabels{1})

osc_1 = repmat(Xlabels(1),18,1);
osc_2 = repmat(Xlabels(2),36,1);
osc_3 = repmat(Xlabels(3),98,1);
osc_4 = repmat(Xlabels(4),212,1);
osc_5 = repmat(Xlabels(5),425,1);
osc_6 = repmat(Xlabels(6),289,1);
osc_7 = repmat(Xlabels(7),98,1);
osc_8 = repmat(Xlabels(8),74,1);


CONDS = cat(1,osc_1,osc_2);

CONDS = cat(1,osc_1,osc_2,osc_3,osc_4,osc_5,osc_6,osc_7,osc_8);

figure; violinplot(RT_ISO_TD, CONDS)


%% Performance (Hit RT) - Phase corr

%     figure; sgtitle(CHAN')
%     edges = [-pi,-3/4*pi,-pi/2, -pi/4,0,pi/4,pi/2,3/4*pi];
%     for k = 1:length(Phase)
%         clear RT_bins
%         RTmat = cell2mat(RTsel{k}');
%         PhaseMat = cell2mat(Phase{k});
%         %bins = discretize(PhaseMat,edges);
%         [Y,E] = discretize(PhaseMat,8);
%         ord = unique(sort(Y));
%         ord = ord(isnan(ord)~=1);
%         for j = 1:length(ord)
%             RT_bins{j} = RTmat(Y==ord(j),1);
%         end
%         subplot(1,8,k); 
%         [h,L,MX,MED,bw] = violin(RT_bins);
%         bins_means= MX;
%         bins_med = MED;
%         hold on; plot(1:8,[bins_means(1),bins_means, bins_means(end)] ,'k')
%         hold on; plot(1:8,[bins_med(1),bins_med, bins_med(end)],'r')
%         [rho(k) pval(k)] = circ_corrcl(E, MX)
%     end
    

%% Save PLV subject*averages data as table
%save resting state long data
%restingLongPLV = PLV;
%save('restingLongPLV','PLV');

%save regular resting state data
% PLVsubjects = PLV;
% PLVresting.avg = PLVsubjects
% PLVresting.name = names;
% PLVresting.frequencies  = frequencies;
% save('restingPLVall','PLVresting');
% 
% PLVresting.avg([9,10]) = PLV; %adding the 2 long Rest state to the 8 conditions
cd('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Data Structures')
load restingPLVall.mat 
PLV = PLVresting.avg; 
frequencies = PLVresting.frequencies;

%% polar hists of phases for each stimulus timing
  clear Phase_deg Phase_rad
  figure(100)
  figure(200)
for ii = 1:length(Phase)
    for jj = 1:length(Phase{ii}) 
        %for kk = 1:length(Phase{ii}{jj})
            Phase_deg{ii}{jj} = rad2deg(Phase{ii}{jj}); %average phase, per participant, across trials
            Phase_rad{ii}{jj} = deg2rad(Phase_deg{ii}{jj}); % the same as above, in rads.
            Phase_rad_group{ii}(jj) = circ_mean(Phase_rad{ii}{jj}');
    end
         figure(100); subplot(4,2,ii);
         pax = polarhistogram(cell2mat(Phase_rad{ii}),15,'Normalization','pdf'); %across all trials pooled together
         rlim([0 0.5])
         hold on; 
         figure(200); subplot(4,2,ii)
         pax = polarhistogram(Phase_rad_group{ii},10,'Normalization','pdf'); %across all trials pooled together
         rlim([0 1.2])
         hold on;
         %Phase_allStim_trials{jj} = cat(1,Phase{1,jj},Phase{2,jj},Phase{3,jj},Phase{4,jj});
end


    
%% plotting circular plots with RT 
clear PhaseQ PhaseQ_deg QQ
figure;
Group_Cond = [cell2mat(RTsel{2}'), cell2mat(Phase{2})'];
Group_Cond(:,2) = [];
Q = quantile(Group_Cond(:,1),3);
Q = cat(2,[0,Q,max(Group_Cond(:,1))]);
for ii = 2:length(Q)
    FF = find(Group_Cond(:,1) <= Q(ii) & Group_Cond(:,1) >= Q(ii-1));
    PhaseQ = Group_Cond(FF,2);
    Phase_deg = rad2deg(PhaseQ);
    Neg = find(Phase_deg<0);
    Phase_deg(Neg) =  Phase_deg(Neg)+360;
    PhaseQ_deg{ii} = Phase_deg;
    H =  histogram(PhaseQ_deg{ii},8);
    Hist{ii} = H.Values;
    QQ(ii,:) = quantile(PhaseQ_deg{ii},8);
end
PhaseQ(1) = [];
PhaseQ_deg(1) = [];
Hist(1) = [];
QQ(1,:) = [];


%% Plotting circular histograms using circ statistics toolbox. By Philipp Berens, 2009
% berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
%Note: SB modified circ_plot to have length of vector [0 0.3].

figure;
titles = {'ISO TD','ISO ASD', 'Jitter Small TD', 'Jitter Small ASD','Jitter Large TD','Jitter Large ASD', 'RAND TD','RAND ASD'};
subplot(4,2,1)
rmax = .28;
% polar(0,rmax,'-k')
% ax = polaraxes;
[a, phi(1),zm(1)] = circ_plot(cell2mat(Phase_rad{1})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{1}); set(gca,'fontsize', 14);
subplot(4,2,2)
[a, phi(2),zm(2)] = circ_plot(cell2mat(Phase_rad{2})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{2}); set(gca,'fontsize', 14);
subplot(4,2,3)
[a, phi(3),zm(3)] = circ_plot(cell2mat(Phase_rad{3})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{3}); set(gca,'fontsize', 14);
subplot(4,2,4)
[a, phi(4),zm(4)] = circ_plot(cell2mat(Phase_rad{4})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{4}); set(gca,'fontsize', 14);
subplot(4,2,5)
[a, phi(1),zm(5)] = circ_plot(cell2mat(Phase_rad{5})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{5}); set(gca,'fontsize', 14);
subplot(4,2,6)
[a, phi(1),zm(6)] = circ_plot(cell2mat(Phase_rad{6})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{6}); set(gca,'fontsize', 14);
subplot(4,2,7)
[a, phi(1),zm(7)] = circ_plot(cell2mat(Phase_rad{7})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{7}); set(gca,'fontsize', 14);
subplot(4,2,8)
[a, phi(1),zm(8)] = circ_plot(cell2mat(Phase_rad{8})','hist',[],10,true,true,'linewidth',2,'color','r');  title(titles{8}); set(gca,'fontsize', 14);

%% calculating mean resultant vector length and Rayleigh test

for ii = 1:length(Phase_rad)
    r(ii) = circ_r(cell2mat(Phase_rad{ii})');
    p_alpha(ii) = circ_rtest(cell2mat(Phase_rad{ii})');
    p_alpha_g(ii) = circ_rtest(Phase_rad_group{ii}');
end


% calculate the vector length for each participant, to make anova test
R_group=[];
for ii = 1:length(Phase_rad)
    for jj = 1:length(Phase_rad{ii})
       R_group{ii}(jj) = circ_r(Phase_rad{ii}{jj}');
    end
end


%% Anova for Phase rose plots - per participant

%all conditions
Y = cat(2,R_group{1}, R_group{2},R_group{3}, R_group{4},R_group{5}, R_group{6},R_group{7},R_group{8}); 
g1 = [ones(1,length(R_group{1})),ones(1,length(R_group{2}))*2,ones(1,length(R_group{3})),ones(1,length(R_group{4}))*2,ones(1,length(R_group{5})),ones(1,length(R_group{6}))*2,...
    ones(1,length(R_group{7})),ones(1,length(R_group{8}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(R_group{1})),ones(1,length(R_group{2})),ones(1,length(R_group{3}))*2,ones(1,length(R_group{4}))*2,...
    ones(1,length(R_group{5}))*3,ones(1,length(R_group{6}))*3,ones(1,length(R_group{7}))*4,ones(1,length(R_group{8}))*4];%condition 1-ISO 2-jit s 3-jit l 4-rand
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])


%iso and rand
Y = cat(2,R_group{1}, R_group{2},R_group{7},R_group{8}); 
g1 = [ones(1,length(R_group{1})),ones(1,length(R_group{2}))*2,ones(1,length(R_group{7})),ones(1,length(R_group{8}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(R_group{1})),ones(1,length(R_group{2})),ones(1,length(R_group{7}))*2,ones(1,length(R_group{8}))*2];%condition 1-ISO 2-jit s 3-jit l
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

%iso and JitS
Y = cat(2,R_group{1}, R_group{2},R_group{3},R_group{4}); 
g1 = [ones(1,length(R_group{1})),ones(1,length(R_group{2}))*2,ones(1,length(R_group{3})),ones(1,length(R_group{4}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(R_group{1})),ones(1,length(R_group{2})),ones(1,length(R_group{3}))*2,ones(1,length(R_group{4}))*2];%condition 1-ISO 2-jit s 3-jit l
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

%% Kuiper test for cdf of samples distributions
[pval, k, K] = circ_kuipertest(Phase_deg_sum_rad{6}, Phase_deg_sum_rad{13}, 100, 1)

%% Watson-Williams test for equality of mean directions (relevant only if the phase itself is important). 
[pval table] = circ_wwtest(Phase_deg_sum_rad{6}, Phase_deg_sum_rad{13})

%% resultant vector length should be > 0.7
[pval, f] = circ_ktest(Phase_deg_sum_rad{1}, Phase_deg_sum_rad{8})

%% [pval, stats] = circ_hktest(alpha, idp, idq, inter, fn)

alpha1 = Phase_deg_sum_rad{1};
alpha2 = Phase_deg_sum_rad{8};
[pval, stats] = circ_hktest(cat(2,alpha1,alpha2),...
    [1:length(alpha1)], [length(alpha1)+1:length(alpha1)+length(alpha2)], 1, {'TD','ASD'})

%%
[rho pval] = circ_corrcc(alpha1, alpha2)    




%% plot ITPC (no SD) - active state

%normalized by subtracting the first point
%1-iso, 2-interchanging, 3-gradual, 4-rand, 5-pseudorand, 6-Jitter s,
%7-Jitter l.        
% 1,5,2,6,3,7,4,8 - resting state load the existing dataset (don't run PLV
% again - use line 745 here).

% means across participants, per condition
figure; 
PLV1 = cell2mat(PLV{1}'); mPLV1 = mean(PLV1,1);         
plot(frequencies, mPLV1,'Color',COLORS(3,:),'LineWidth',2.5);
pause
PLV2 = cell2mat(PLV{2}'); mPLV2 = mean(PLV2,1);
hold on; plot(frequencies, mPLV2,'Color',COLORS(10,:),'LineWidth',2.5);
pause
PLV3 = cell2mat(PLV{3}'); mPLV3 = mean(PLV3,1);
hold on; plot(frequencies, mPLV3,'Color',COLORS(4,:),'LineWidth',2.5);
pause
PLV4 = cell2mat(PLV{4}'); mPLV4 = mean(PLV4,1);
hold on; plot(frequencies,mPLV4,'Color',COLORS(11,:),'LineWidth',2.5);
pause
PLV5 = cell2mat(PLV{5}'); mPLV5 = mean(PLV5,1);
hold on; plot(frequencies, mPLV5,'Color',COLORS(6,:),'LineWidth',2.5);
pause
PLV6 = cell2mat(PLV{6}'); mPLV6 = mean(PLV6,1);
hold on; plot(frequencies, mPLV6,'Color',COLORS(13,:),'LineWidth',2.5);
pause
PLV7 = cell2mat(PLV{7}'); mPLV7 = mean(PLV7,1);
hold on; plot(frequencies, mPLV7,'Color',COLORS(7,:),'LineWidth',2.5);
pause
PLV8 = cell2mat(PLV{8}'); mPLV8 = mean(PLV8,1);
hold on; plot(frequencies, mPLV8,'Color',COLORS(14,:),'LineWidth',2.5);
pause

% adding long resting states
figure; title('Long Resting State (30 sec)')
PLV9 = cell2mat(PLV{9}'); mPLV9 = mean(PLV9,1);
hold on; plot(frequencies, mPLV9,'Color','k','LineWidth',2.5);
PLV10 = cell2mat(PLV{10}'); mPLV10 = mean(PLV10,1);
hold on; plot(frequencies, mPLV10,'Color','m','LineWidth',2.5);



xlabel('Frequency (Hz)');
ylabel('Coherence (AU)');
%xlim([1 2.5]);
%ylim([0.1 0.3]);
 
legend('TD ISO','ASD ISO','TD Jitter S','ASD Jitter S','TD Rand','ASD Rand');

title('Phase locking values for all conditions')

set(gca,'fontsize', 14);

% For long resting state (before the experiment): 
% figure; 
% PLV9 = cell2mat(PLV{9}'); mPLV1 = mean(PLV9,1);         
% plot(frequencies, mPLV1,'Color',COLORS(3,:),'LineWidth',2.5);
% PLV10 = cell2mat(PLV{10}'); mPLV2 = mean(PLV10,1);
% hold on; plot(frequencies, mPLV2,'Color',COLORS(10,:),'LineWidth',2.5);
% 


%% Save PLV subject*averages data as table
%save resting state long data
restingLongPLV = PLV;
save('restingLongPLV','PLV');

%save regular resting state data
PLVsubjects = PLV;
PLVresting.avg = PLVsubjects
PLVresting.name = names;
PLVresting.frequencies  = frequencies;
save('restingPLV','PLVresting');


%% plot individual subjects PLV
%looks better on CP1,CP2,CPz or C1,Cz,C2
figure;
conds = [6,13];
for ii = 1:length(conds)
    for jj = 1:length(PLV{conds(ii)})
        subplot(1,2,ii);
        plot(frequencies, PLV{conds(ii)}{jj},'LineWidth',1.5);
        hold on;
    end
    xlim([1 2.5]);
    ylim([0 3.5]);
    set(gca,'fontsize', 14);
    xlabel('Frequency (Hz)');
    ylabel('Coherence (AU)');
end

%% plot PLV with bounded lines
%looks better on CP1,CP2,CPz or C1,C2,Cz
PLVsel = {PLV1,PLV2,PLV3,PLV4,PLV5,PLV6,PLV7,PLV8}; %selected conds
%PLVsel = {PLV1,PLV2}; %selected conds
%PLVsel = {PLV9,PLV10}; %selected conds


% 1 plot per condition
figure;
plotOrd = [1,1,2,2,3,3,4,4];

ColOrd = [3,10,4,11,6,13,7,14];
for ii = 1:length(PLVsel)
   
    %for jj = 1:length(PLV{ii})
    [mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(PLVsel{ii}),std(PLVsel{ii})./sqrt(size(PLVsel{ii},1)),FS,frequencies); % Draw bounded lines
    %[X,mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(cell2mat(PLV{ii}')),std(cell2mat(PLV{ii}'))./sqrt(length(PLV{ii})),FS,x); % Draw bounded lines
    subplot(1,4,plotOrd(ii));
    
    set(gca, 'XScale', 'log')
    shadedErrorBar(frequencies,mean_smooth,error_smooth,{COLORS(ColOrd(ii),:),'LineWidth',1},0);

    hold on;
    box off
    xlim([1 6])
    ylim([0 0.5])

end

%legend('TD Cue','TD No Cue','ASD Cue','ASD No Cue');
xlabel('Frequency (Hz)');
ylabel('Coherence (AU)');
title('Mean Phase locking values for all conditions')

% one plot per group - all condition per group.
PLVsel = {PLV1,PLV3,PLV5,PLV7,PLV2,PLV4,PLV6,PLV8}; %selected conds
%PLVsel = {PLV{1},PLV{3},PLV{5},PLV{7},PLV{2},PLV{4},PLV{6},PLV{8}}; %selected conds

figure;
plotOrd = [1,1,1,1,2,2,2,2];
ColOrd = [3,4,6,7,10,11,13,14];

for ii = 1:length(PLVsel)
   
    %for jj = 1:length(PLV{ii})
    [mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(PLVsel{ii}),std(PLVsel{ii})./sqrt(size(PLVsel{ii},1)),FS,frequencies); % Draw bounded lines
    %[X,mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(cell2mat(PLV{ii}')),std(cell2mat(PLV{ii}'))./sqrt(length(PLV{ii})),FS,x); % Draw bounded lines
    subplot(1,2,plotOrd(ii));
    set(gca, 'XScale', 'log')
    shadedErrorBar(frequencies,mean_smooth,error_smooth,{COLORS(ColOrd(ii),:),'LineWidth',1},0);

    hold on;
    box off
    xlim([1.1 6])
    ylim([0 0.5])
   
end
%% Statistics
%rank test to test statistical significance in the PLVs

x = mean(cell2mat(PLV{1}'));
y = mean(cell2mat(PLV{5}'));
%IF = 15:27; %interesting freqs
IF = 19:24;
[p,h,stats] = ranksum(x,y)

% permutation test on two groups
clear PLVmeans PLVmax
for m = 1:8; %interesting conds - iso and jitter
    for n = 1:length(PLV{m})
        PLVmeans{m}(n) = mean(PLV{m}{n}(IF));
        PLVmax{m}(n) = max(PLV{m}{n}(IF));
        PLVmed{m}(n) = median(PLV{m}{n}(IF));
    end
end

%permutation test between groups (cue only)
[p, observeddifference, effectsize] = permutationTest(PLVmax{4}, PLVmax{8}, 10000,  'sidedness','larger','plotresult',1)

%max value
%iso
[p, observeddifference, effectsize] = permutationTest(PLVmax{2}, PLVmax{6}, 10000,  'sidedness','larger','plotresult',1)
%jitter s
[p, observeddifference, effectsize] = permutationTest(PLVmax{6}, PLVmax{13}, 10000,  'sidedness','larger','plotresult',1)
%jitter l
[p, observeddifference, effectsize] = permutationTest(PLVmax{10}, PLVmax{22}, 10000,  'sidedness','larger','plotresult',1)
%rand
[p, observeddifference, effectsize] = permutationTest(PLVmax{4}, PLVmax{11}, 10000,  'sidedness','larger','plotresult',1)


%% plot differences 
%raincloud


ITPCmaxVals = xlsread('ITPCFreqMax.xlsx');
plotITPC{1,1} = PLVmax{1};
plotITPC{1,2} = PLVmax{8};
plotITPC{2,1} = PLVmax{6};
plotITPC{2,2} = PLVmax{13};
plotITPC{3,1} = PLVmax{7};
plotITPC{3,2} = PLVmax{14};
plotITPC{4,1} = PLVmax{4};
plotITPC{4,2} = PLVmax{11};

figure;
cl = rand(2,3);
h   = rm_raincloud(plotITPC, cl);


%2-way anova interaction  betweem max value at the area of the ITPC peak

Y = cat(2,PLVmax{1}, PLVmax{8},PLVmax{2}, PLVmax{13},PLVmax{7}, PLVmax{14}); 
g1 = [ones(1,length(PLVmax{1})),ones(1,length(PLVmax{8}))*2,ones(1,length(PLVmax{6})),ones(1,length(PLVmax{13}))*2,ones(1,length(PLVmax{7})),ones(1,length(PLVmax{14}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(PLVmax{1})),ones(1,length(PLVmax{8})),ones(1,length(PLVmax{6}))*2,ones(1,length(PLVmax{13}))*2,...
    ones(1,length(PLVmax{7}))*3,ones(1,length(PLVmax{14}))*3];%condition 1-ISO 2-jit s 3-jit l
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])


st_boxdotplot([1:8],{PLVmeans{1}; PLVmeans{13};PLVmeans{11}; PLVmeans{23}; PLVmeans{12}; PLVmeans{24}...
   ;PLVmeans{10}; PLVmeans{22}} ,[0,0,0;0.2,0.8,0.5;0.5,0.1,0.9;0.4,0.2,0.2;0,0,0;0.2,0.8,0.5;0.5,0.1,0.9;0.4,0.2,0.2...
   ],'iqr',[],[],[],[],80,[],[],1);

%g1=1;g2=1: TD ISO
%g1=2;g2=1: ASD ISO
%g1=1; g2=2: TD Rand
%g1=2; g2=2: ASD Rand
% TD Rand is diff than ASD Rand


%% Anova for the resting state PLV groups

Y = cat(2,PLVmeans{1}, PLVmeans{5},PLVmeans{2}, PLVmeans{6});   
g1 = [ones(1,length(PLVmeans{1})),ones(1,length(PLVmeans{5}))*2,ones(1,length(PLVmeans{2})),ones(1,length(PLVmeans{6}))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(PLVmeans{1})),ones(1,length(PLVmeans{5})),ones(1,length(PLVmeans{2}))*2,ones(1,length(PLVmeans{6}))*2];
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

st_boxdotplot([1:4],{PLVmeans{1}; PLVmeans{5};PLVmeans{2}; PLVmeans{6}},[0,0,0;0.2,0.8,0.5;0.5,0.1,0.9;0.4,0.2,0.2],'iqr',[],[],[],[],80,[],[],1);

%% Anova for the resting state PLV groups (iso and rand), for all points in the trial (to create a significane bar above the plot)
clear P;
for point = 1:length(PLV{1}{1})
    
    Y = cat(1,PLV1(:,point), PLV2(:,point),PLV5(:,point), PLV6(:,point))';   
    g1 = [ones(1,size(PLV1,1)),ones(1,size(PLV2,1))*2,ones(1,size(PLV5,1)),ones(1,size(PLV6,1))*2];%group 1-TD 2-ASD
    g2 =  [ones(1,size(PLV1,1)),ones(1,size(PLV2,1)),ones(1,size(PLV5,1))*2,ones(1,size(PLV6,1))*2];
    
    p = anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'});
    P(:,point) = p;
end


groupEffect = find(P(1,:)<0.05);
condEffect = find(P(2,:)<0.05);
interactionEffect = find(P(3,:)<0.05);

%% %%%%%%%%% normalizing rest plv to the total rest (long) plv
mPLV1 = mPLV1./mPLV9;
mPLV2 = mPLV2./mPLV9;
mPLV3 = mPLV3./mPLV9;
mPLV4 = mPLV4./mPLV9;
mPLV5 = mPLV5./mPLV10;
mPLV6 = mPLV6./mPLV10;
mPLV7 = mPLV7./mPLV10;
mPLV8 = mPLV8./mPLV10;



%% plot resting state conditions
% 1,5,2,6,3,7,4,8 - resting state (NOTE: ITS NOT NORMALIZED)

figure; 
subplot(1,2,1); 
plot(frequencies, mPLV1); hold on;
plot(frequencies, mPLV2); hold on; 
plot(frequencies, mPLV3); hold on; 
plot(frequencies, mPLV4);
set(gca, 'XScale', 'log')
ylim([0.18 0.33])
xlim([0.8 10])
hold on;
line(frequencies(condEffect),ones(1,length(condEffect))*0.32)
hold on; 
line(frequencies(interactionEffect),ones(1,length(interactionEffect))*0.323,'Color','k')

subplot(1,2,2); 
plot(frequencies, mPLV5); hold on; 
plot(frequencies, mPLV6); hold on; 
plot(frequencies, mPLV7); hold on; 
plot(frequencies, mPLV8);
set(gca, 'XScale', 'log')
ylim([0.18 0.33])
xlim([0.8 10])

%plot only the 2 conds for each group
figure; 
plot(frequencies, mPLV1,'k'); hold on;
plot(frequencies, mPLV2,'k'); hold on;
plot(frequencies, mPLV5,'r'); hold on; 
plot(frequencies, mPLV6,'r');

set(gca, 'XScale', 'log')
ylim([0.18 0.33])
xlim([0.8 10])
title('Interaction effects')

hold on; 
line(frequencies(interactionEffect),ones(1,length(interactionEffect))*0.19,'Color','k')