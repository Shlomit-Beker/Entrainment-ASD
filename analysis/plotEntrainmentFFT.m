% Main code to calc plot and do stats on FFTbfor Entrainment paper (Fig 2)
%CHAN = {'FCz','FC2'};
%CHAN = {'C1','C3','P1','Pz','Cz','C2','C4','C6','T8'};
%CHAN = {'O1','Oz','Iz','P10','P9'};
% load('ERP_visual_long(5.5sec)');
% load('grandAvg_vis');

CHAN = {'O1','Oz','P9','P10'};

%CHAN = {'O1','Oz'};

 %elec = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2}))];
 elec = [find(strcmp(grandAvg{1}.label,CHAN{1})),find(strcmp(grandAvg{1}.label,CHAN{2})),...
   find(strcmp(grandAvg{1}.label,CHAN{3})), find(strcmp(grandAvg{1}.label,CHAN{4}))];
clear norm_data spec_data Nspec_data spec_data entrainFft NentrainFft 


%   ERP{1}(14) = [];
% %  ERPb{6}(13) = [];
%  ERP{7}(8) = [];
%  ERP{11}(11) = [];
% % ERPb{4}(2) = [];

% ERP{6}(7) = [];



FS = 256;


for i = 1:length(ERP)   %cond
    for m = 1:64
        for j = 1:length(ERP{i}) %participant
            [entrainSpec,f] = pwelch(ERP{i}{j}.avg(m,:),1400,1300,0.5:0.1:10,FS); % pwelch(mean(data{i}{j}(elec,:),1),700,600,0:0.1:7,Fs);
            %[entrainSpec,f] = pwelch(mean(ERP{i}{j}.avg(elec,:),1),1400,1300,0.5:0.1:10,Fs); % pwelch(mean(data{i}{j}(elec,:),1),700,600,0:0.1:7,Fs);
            %norm_data{i}(j,:) = (entrainSpec - min(entrainSpec));% / ( max(entrainSpec) - min(entrainSpec));
            spec_data{i}(m,:,j) = entrainSpec;
            IDs{i}(j) = str2num(ERP{i}{j}.name);
        end

        entrainFft{i}(m,:) = mean(spec_data{i}(m,:,:),3);
    end

end

%% mat (subjects * freqs) for specific channels
figure
for i = 1:length(spec_data)
    spec_data_indiv{i} = squeeze(mean(spec_data{i}(elec,:,:),1));
    
end
figure
flag = 0;
for i = [1,8,6,13,7,14,4,11]
    flag = flag+1;
    subplot(4,2,flag)
    
    plot(f,spec_data_indiv{i},'Color',[0.5,0.5,0.5])
    hold on; plot(f, mean(spec_data_indiv{i},2),'k','LineWidth',1)
    xlim([1 4]);
    ylim([-0.5 4.5])
end


%% put in FT structure for plotting
%load('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis/64_lay.mat')
%load('/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis/64_lay.mat')
load('/Users/shlomitbeker/Library/CloudStorage/OneDrive-TheMountSinaiHospital/A ENTRAINMENT AND OSCILLATION IN ASD/Analysis/64_lay.mat')
relevantChannels = ERP{1}{1}.label(1:64);%{'PO3','POz','PO4','Oz','O1','O2','AF3','AFz','AF4','F1','Fz','F2'};
CHANNELS = find(ismember(lay.label,relevantChannels)==1);

lay.pos = lay.pos(CHANNELS,:);
lay.width = lay.width(CHANNELS,:);
lay.height = lay.height(CHANNELS,:);
lay.label = lay.label(CHANNELS);


meanSPCTRM1.label=lay.label;
meanSPCTRM1.powspctrm = entrainFft{1}; %note! the matrix should be chans*freq
meanSPCTRM1.dimord='chan_freq';
meanSPCTRM1.freq = f;
meanSPCTRM1.fsample = FS;

meanSPCTRM8.label=lay.label;
meanSPCTRM8.powspctrm = entrainFft{8}; %note! the matrix should be chans*freq
meanSPCTRM8.dimord='chan_freq';
meanSPCTRM8.freq = f;
meanSPCTRM8.fsample = FS;

cfg=[];
cfg.interactive='yes';
cfg.layout = lay;
cfg.xlim = [0.5 2]; %frequencies(1:20);%[0.1 3];
cfg.clim = [0 4];
cfg.parameter='powspctrm';
cfg.showlabels    = 'yes';

figure;ft_multiplotER(cfg,meanSPCTRM1,meanSPCTRM8); set(gca,'fontsize',16); 
%figure;ft_topoplotER(cfg,meanSPCTRM6,meanSPCTRM13);  set(gca,'fontsize',16); 

%%
CONDS = [1,6,7,4];
%flag = 0;
flagPic = -1;
figure; 
%spec_data_indiv{1}(:,14) = [];
for m = 1:4

    %flag = flag+1;
    flagPic = flagPic+2;
    %signal = entrainFft(flag);
    
    subplot(4,2,flagPic)
    
    plot(grandAvg{CONDS(m)}.time, max(grandAvg{CONDS(m)}.avg(elec,:)),'k');
    hold on; 
    plot(grandAvg{CONDS(m)+7}.time, max(grandAvg{CONDS(m)+7}.avg(elec,:)),'r');
    xlim([-0.5 5]);
    ylim([-10 10]);
    box off
    %plot(f,entrainFft(flag),'k'); hold on; plot(f,entrainFft(13,:));
    subplot(4,2,flagPic+1)
    entrainFft_error = std(spec_data_indiv{CONDS(m)}')./sqrt(length(ERP{CONDS(m)}));
    %[X,mean_smooth, error_smooth] = drawBoundedLines_NEW(entrainFft(CONDS(m),:),entrainFft_error,FS,f); % Draw bounded lines
    %ax1 = shadedErrorBar(X,mean_smooth,error_smooth,{'k','LineWidth',0.5},0);
    [mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(entrainFft{CONDS(m)}(elec,:)),entrainFft_error,FS,f); % Draw bounded lines
    ax1 = shadedErrorBar(f, mean_smooth,error_smooth,{'k','LineWidth',0.5},0);
  

    hold on;
    entrainFft_error = std(spec_data_indiv{CONDS(m)+7}')./sqrt(length(ERP{CONDS(m)+7}));
    [mean_smooth, error_smooth] = drawBoundedLines_NEW(mean(entrainFft{CONDS(m)+7}(elec,:)),entrainFft_error,FS,f); % Draw bounded lines
    ax1 = shadedErrorBar(f,mean_smooth,error_smooth,{'r','LineWidth',0.5},0);
    ylim([0 3])

    xlim([1 5]);
    box off
    
end

%% Stats
win = [8:12];
for i = [CONDS, CONDS+7]
    for j = 1:size(spec_data_indiv{i},2)
        peakComp{i}(j) = mean(spec_data_indiv{i}(win,j));
    end
end
[p, h, stats] = ranksum(peakComp{1},peakComp{8})
[p, h, stats] = ranksum(peakComp{6},peakComp{13})
[p, h, stats] = ranksum(peakComp{7},peakComp{14})
[p, h, stats] = ranksum(peakComp{4},peakComp{11})

% Mick's plot
colors = rand(2,3);
figure; subplot(1,4,1)
st_boxdotplot([1:2],{peakComp{1},peakComp{8}}...
    ,colors,'iqr',[],[],[],0.5,30,0.5,[],1);  %mick RT plot func
title('Iso');
set(gca, 'YScale', 'log')
ylim([0.006 12])

subplot(1,4,2)
st_boxdotplot([1:2],{peakComp{6},peakComp{13}},...
    colors,'iqr',[],[],[],0.5,30,0.5,[],1);
title('Small Jitter');
set(gca, 'YScale', 'log')
ylim([0.006 12])

subplot(1,4,3)
st_boxdotplot([1:2],{peakComp{7},peakComp{14}},colors,'iqr',[],[],[],0.5,30,0.5,[],1);  %mick RT plot func
title('Large Jitter');
set(gca, 'YScale', 'log')
ylim([0.006 12])


subplot(1,4,4)
st_boxdotplot([1:2],{peakComp{4},peakComp{11}},colors,'iqr',[],[],[],0.5,30,0.5,[],1);  %mick RT plot func
title('Large Jitter');
set(gca, 'YScale', 'log')
ylim([0.006 12])

% Anova on peakComp - added 2/14/25

Y = [peakComp{1},peakComp{6},peakComp{7},peakComp{4},peakComp{8},peakComp{13},peakComp{14},peakComp{11}];
g1 = [ones(1,length(peakComp{1})),ones(1,length(peakComp{6})),ones(1,length(peakComp{7})),ones(1,length(peakComp{4})),ones(1,length(peakComp{8}))*2,ones(1,length(peakComp{13}))*2,...
    ones(1,length(peakComp{14}))*2,ones(1,length(peakComp{11}))*2];%group 1-TD 2-ASD

g2 =  [ones(1,length(peakComp{1})),ones(1,length(peakComp{6}))*2,ones(1,length(peakComp{7}))*3,ones(1,length(peakComp{4}))*4,ones(1,length(peakComp{8})),ones(1,length(peakComp{13}))*2,...
    ones(1,length(peakComp{14}))*3,ones(1,length(peakComp{11}))*4];% conds - 1(iso) 2(jit s) 3(jit l) 4(rand)
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])


% permutation test on FFT spectra - added 3/25/25
[p, observeddifference, effectsize] = permutationTest(peakComp{1}, peakComp{8}, 10000,  'sidedness','larger','plotresult',1)
[p, observeddifference, effectsize] = permutationTest(peakComp{6}, peakComp{13}, 10000,  'sidedness','larger','plotresult',1)
[p, observeddifference, effectsize] = permutationTest(peakComp{7}, peakComp{14}, 10000,  'sidedness','larger','plotresult',1)
[p, observeddifference, effectsize] = permutationTest(peakComp{4}, peakComp{11}, 10000,  'sidedness','larger','plotresult',1)

%% descriptive stats
meanPeak_TD = [mean(peakComp{1}) mean(peakComp{6}) mean(peakComp{7}) mean(peakComp{4})]

stdPeak_TD = [std(peakComp{1})./sqrt(19) std(peakComp{6})./sqrt(18) std(peakComp{7})./sqrt(17) std(peakComp{4})./sqrt(19)]

meanPeak_ASD = [mean(peakComp{8}) mean(peakComp{13}) mean(peakComp{14}) mean(peakComp{11})]
stdPeak_ASD = [std(peakComp{8})./sqrt(18) std(peakComp{13})./sqrt(17) mean(peakComp{14})./sqrt(17) std(peakComp{11})./sqrt(18)]


%% Save Variables
FFTpeaks.amp = peakComp;
FFTpeaks.names = IDs;

save('FFTpeaks','FFTpeaks');


%% ploting fooof

TD_ISO = [2.33	1.32	1.143	1.36	1.75	1.51	1.85	2.03	1.38	2.07	2.97	1.46	2.04	1.4	1.15	2.9	2.45	1.2];
TD_SJ = [0.77	1.32	1.5	2.12	1.25	1.08	2.26	2.14	1.42	1.56	2.11	1.38	0.96	1.82	1.07];
TD_LJ = [0.89	1.17	0.44	0.72	0.797	0.71	1.037	0.974	0.62	1.44	1.05	0.71	0.98	1.044	0.97];

ASD_ISO = [1.68	1.12	0.99	1.08	0.95	1.96	1.22	1.01	1.19	2.03	1.78	1.35	1.123	1.88	1.99	2.36];
ASD_SJ = [1.23	1.35	1.14	1.57	1.3	1.91	1.66	1.46	1.4	1.17	1.84	1.68	0.79	1.45	1.57	0.8];
ASD_LJ = [1.07	0.83	1.288	1.21	0.62	0.91	0.92	1.16];

%% raincloud
data{1,1} = TD_ISO;
data{1,2} = ASD_ISO;
data{2,1} = TD_SJ;
data{2,2} = ASD_SJ;
data{3,1} = TD_LJ;
data{3,2} = ASD_LJ;

% plot (different style)
fig_position = [200 200 600 400]; % coordinates for figures
cl = rand(2,3);
f9  = figure('Position', fig_position);
h   = rm_raincloud(data, cl);
title(['Figure M9' newline 'Repeated measures raincloud plot']);

%anova

Y = [TD_ISO,TD_SJ,TD_LJ,ASD_ISO,ASD_SJ,ASD_LJ];
g1 = [ones(1,length(TD_ISO)),ones(1,length(TD_SJ))*2,ones(1,length(TD_LJ)),ones(1,length(ASD_ISO))*2,ones(1,length(ASD_SJ)),ones(1,length(ASD_LJ))*2];%group 1-TD 2-ASD
g2 =  [ones(1,length(TD_ISO)),ones(1,length(ASD_ISO)),ones(1,length(TD_SJ))*2,ones(1,length(ASD_SJ))*2,...
    ones(1,length(TD_LJ))*3,ones(1,length(ASD_LJ))*3];%condition 1-ISO 2-jit s 3-jit l
[~,~,stats]= anovan(Y,{g1 g2},'model','interaction','varnames',{'g1','g2'})
figure; 
results = multcompare(stats,'Dimension',[1 2])

%% raincloud
data{1,1} = peakComp{1};
data{1,2} = peakComp{8};
data{2,1} = peakComp{6};
data{2,2} = peakComp{13};
data{3,1} = peakComp{7};
data{3,2} = peakComp{14};
data{4,1} = peakComp{4};
data{4,2} = peakComp{11};

% plot
fig_position = [200 200 600 400]; % coordinates for figures
cb = rand(2,3);

f7 = figure('Position', fig_position);
subplot(1, 4, 1)
h1 = raincloud_plot(data{1,1}, 'box_on', 1, 'color', cb(1,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .15, 'dot_dodge_amount', .15,...
     'box_col_match', 0);
h2 = raincloud_plot(data{1,2}, 'box_on', 1, 'color', cb(2,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .35, 'dot_dodge_amount', .35, 'box_col_match', 0);
legend([h1{1} h2{1}], {'Group 1', 'Group 2'});
title(['Figure M7' newline 'A) Dodge Options Example 1']);
%set(gca,'XLim', [0 40], 'YLim', [-.075 .15]);
box off

subplot(1, 4, 2)
h1 = raincloud_plot(data{2,1}, 'box_on', 1, 'color', cb(1,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .15, 'dot_dodge_amount', .15,...
     'box_col_match', 0);
h2 = raincloud_plot(data{2,2}, 'box_on', 1, 'color', cb(2,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .35, 'dot_dodge_amount', .35, 'box_col_match', 0);
legend([h1{1} h2{1}], {'Group 1', 'Group 2'});
title(['Figure M7' newline 'A) Dodge Options Example 1']);
%set(gca,'XLim', [0 40], 'YLim', [-.075 .15]);
box off

subplot(1, 4, 3)
h1 = raincloud_plot(data{3,1}, 'box_on', 1, 'color', cb(1,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .15, 'dot_dodge_amount', .15,...
     'box_col_match', 0);
h2 = raincloud_plot(data{3,2}, 'box_on', 1, 'color', cb(2,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .35, 'dot_dodge_amount', .35, 'box_col_match', 0);
legend([h1{1} h2{1}], {'Group 1', 'Group 2'});
title(['Figure M7' newline 'A) Dodge Options Example 1']);
%set(gca,'XLim', [0 40], 'YLim', [-.075 .15]);
box off

subplot(1, 4, 4)
h1 = raincloud_plot(data{4,1}, 'box_on', 1, 'color', cb(1,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .15, 'dot_dodge_amount', .15,...
     'box_col_match', 0);
h2 = raincloud_plot(data{4,2}, 'box_on', 1, 'color', cb(2,:), 'alpha', 0.5,...
     'box_dodge', 1, 'box_dodge_amount', .35, 'dot_dodge_amount', .35, 'box_col_match', 0);
legend([h1{1} h2{1}], {'Group 1', 'Group 2'});
title(['Figure M7' newline 'A) Dodge Options Example 1']);
%set(gca,'XLim', [0 40], 'YLim', [-.075 .15]);
box off



% plot (different style)
fig_position = [200 200 600 400]; % coordinates for figures
cl = rand(2,3);
f9  = figure('Position', fig_position);
h   = rm_raincloud(data, cl);
title(['Figure M9' newline 'Repeated measures raincloud plot']);

% save
print(f9, fullfile(figdir, '9RmRain1.png'), '-dpng');




