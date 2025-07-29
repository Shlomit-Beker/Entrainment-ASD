
 
% Phase Locking (Intra-Trial-Phase-Coherence) analysis. Shlomit Beker 2018
% runs on selected channels. 
%% parameters for phase locking
%startup;

clear PLallstim PLV
%CHAN = {'O2'};
CHAN = {'Iz','Oz'};
%CHAN = {'O1','Oz','O2'}; 
%CHAN = {'C1','C2','Cz'}; % for entrainment: 'C1','Cz','C2' for visual sequence: 'AF3','Fp1','AFz','Fpz','A7','PO3','POz','PO4',
%CHAN = {'PO3','POz','PO4',}; 
%CHAN = {'CP1','CP2','CPz'}
%C = [find(strcmp(ERPb{1}{1}.label,CHAN{1}))];

C = [find(strcmp(ERP{1}{1}.label,CHAN{1})),find(strcmp(ERP{1}{1}.label,CHAN{2}))];
%C = [find(strcmp(ERPb{1}{1}.label,CHAN{1})),find(strcmp(ERPb{1}{1}.label,CHAN{2})),find(strcmp(ERPb{1}{1}.label,CHAN{3}))];%...
CHANNELS = C;% **PLV in the VAMP paper was calulated on C (central)channels***Phase is calculated on POz,PO3,PO4*****
SAMP_RATE = 256; 
LOW_FREQUENCY = 0.5;
HIGH_FREQUENCY = 10; %range of frequencies on which to make the coherence
OMEGA = 6;
LENGTH_WIN = length(DATA{1}{1}{1})./SAMP_RATE;   
%LENGTH_WIN = length(DATA{19}{1}{1})./SAMP_RATE;  

TIME_WINDOW = 1:LENGTH_WIN*SAMP_RATE;
COLORS = [inferno(7);viridis(7)];
clear i;
FOI = 21; %location of 1.5Hz in the frequencies vector
%for short trials
start = 51;
STIM_TIMES = start;

%% reref and baseline DATA
if Signal == 6
    DATAa = DATA([1,13,11,23,12,24,9,21]);
    %DATAa = [DATA(1),DATA(13),DATA(11),DATA(23),DATA(12),DATA(24),DATA(9),DATA(21)];
    names = NAMES([1,13,11,23,12,24,9,21]);
elseif Signal == 7
    DATAa = DATA([1,4,6,7,9,13,15,16]);
    names = NAMES([1,4,6,7,9,13,15,16]);
end
%DATAa  = DATA([19,20]); %for the long resting states (from the 30 seconds
%at the beginning of the experiment)
DATAr = rerefData(DATAa,37);

%% Create the data mat files, by condition
CHAN             %spit current Channels 
flag = 0;
clear sumAngles STphase1 Angles Phase PL PLV
    for COND = 1:length(DATAr)
        for stim = 1:length(STIM_TIMES)
            for participant = 1:length(DATAr{COND})
                clear STphase1
                currentData = DATAr{COND}{participant};
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
                PLallstim(:) = PL;
                Phase{stim,COND}{participant} = squeeze(STphase1(FOI,TOI,:))';
                PLV{COND}{participant} = PLallstim;
            
            end     
        end
    end

%% Save PLV subject*averages data as table
%save resting state long data
restingLongPLV = PLV;
save('restingLongPLV','PLV');

%save regular resting state data
PLVsubjects = PLV;
PLVresting.avg = PLVsubjects
PLVresting.name = names;
PLVresting.frequencies  = frequencies;
save('restingPLVall','PLVresting');

PLVresting.avg([9,10]) = PLV; %adding the 2 long Rest state to the 8 conditions

%% polar hists of phases for each stimulus timing
  clear Phase_deg Phase_rad
for ii = 1:size(Phase,1)
    Fig_s = figure(100+ii); title([num2str(ii)]);
    for jj = 1:size(Phase,2)
        for kk = 1:length(Phase{ii,jj})
            Phase_deg{ii,jj}(kk) = meanangle(rad2deg(Phase{ii,jj}{kk})); %average phase, per participant, across trials
            Phase_rad{ii,jj}(kk) = deg2rad(Phase_deg{ii,jj}(kk)); % the same as above, in rads.
        end
         figure(Fig_s); subplot(2,7,jj);
         polarhistogram(Phase_rad{ii,jj},10);
         hold on; 
         %Phase_allStim_trials{jj} = cat(1,Phase{1,jj},Phase{2,jj},Phase{3,jj},Phase{4,jj});
    end
end
    
%% phases across stim, per participant, plus plot
clear Phase_allStimSubject Mat_1 Mat_2
for ii = 1:size(Phase, 2)
   %Fig_s = figure(ii); title([num2str(ii)]);

   Mat_1{ii} = cat(1,Phase{1,ii},Phase{2,ii},Phase{3,ii},Phase{4,ii},Phase{5,ii},Phase{6,ii});
     for jj = 1:size(Mat_1{ii},2)
        Mat_2{ii}{jj} = cell2mat(Mat_1{ii}(1:size(Phase, 1),jj));
        Phase_allStimSubject{ii}{jj} = meanangle(rad2deg(Mat_2{ii}{jj}),1);
%          figure(Fig_s); subplot(5,4,jj);
%          polarhistogram(Phase_allStimSubject{ii}{jj},18);
%          hold on; 
     end
end

%%
clear Phase_deg_sum Phase_deg_sum_rad
Fig_a = figure(200); title('all cues');
Fig_m = figure(300); title('mean cue');
for ii=1:14
        Phase_rad_sum{ii} = cat(2,Phase_rad{1,ii},Phase_rad{2,ii},Phase_rad{3,ii},Phase_rad{4,ii},Phase_rad{5,ii},Phase_rad{6,ii});
        figure(Fig_a); subplot(2,7,ii); polarhistogram(Phase_rad_sum{ii},20,'FaceColor',COLORS(ii,:));
        hold on; 
        %polarplot([0 real(zm(ii))], [0, imag(zm(ii))],'r');
        Phase_deg_sum{ii} = meanangle(cat(1,Phase_deg{1,ii},Phase_deg{2,ii},Phase_deg{3,ii},Phase_deg{4,ii},Phase_deg{5,ii},Phase_deg{6,ii}));
        hold on;
        rlim([0 20])
        %Phase_deg_sum{ii} = meanangle(cat(1,Phase_deg{2,ii},Phase_deg{3,ii},Phase_deg{4,ii}));

        Phase_deg_sum_rad{ii} = deg2rad(Phase_deg_sum{ii});
        figure(Fig_m); subplot(2,7,ii); polarhistogram(Phase_deg_sum_rad{ii},10,'FaceColor',COLORS(ii,:));
        rlim([0 8])
        hold on; 
end
    
%% Plotting circular histograms using circ statistics toolbox. By Philipp Berens, 2009
% berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
figure;
titles = {'ISO TD','ISO ASD', 'RAND TD','RAND ASD','Jitter Small TD', 'Jitter Small ASD','Jitter Large TD','Jitter Large ASD'};
subplot(4,2,1)
[a, phi(1),zm(1)] = circ_plot(Phase_rad{1}','hist',[],10,false,true,'linewidth',2,'color','r');  title(titles{1}); set(gca,'fontsize', 14);
subplot(4,2,2)
[a, phi(2),zm(2)] = circ_plot(Phase_rad{2}','hist',[],10,false,true,'linewidth',2,'color','r');  title(titles{2}); set(gca,'fontsize', 14);
subplot(4,2,3)
[a, phi(3),zm(3)] = circ_plot(Phase_rad{3}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{3}); set(gca,'fontsize', 14);
subplot(4,2,4)
[a, phi(4),zm(4)] = circ_plot(Phase_rad{4}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{4});set(gca,'fontsize', 14); 
subplot(4,2,5)
[a, phi(5),zm(5)] = circ_plot(Phase_rad{5}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{5}); set(gca,'fontsize', 14);
subplot(4,2,6)
[a, phi(6),zm(6)] = circ_plot(Phase_rad{6}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{6});set(gca,'fontsize', 14); 
subplot(4,2,7)
[a, phi(7),zm(7)] = circ_plot(Phase_rad{7}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{7}); set(gca,'fontsize', 14);
subplot(4,2,8)
[a, phi(8),zm(8)] = circ_plot(Phase_rad{8}','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{8});set(gca,'fontsize', 14); 


% figure;
% titles = {'ISO TD','ISO ASD', 'RAND TD','RAND ASD','Jitter Small TD', 'Jitter Small ASD','Jitter Large TD','Jitter Large ASD'};
% subplot(4,2,1)
% [a, phi(1),zm(1)] = circ_plot(cell2mat(Phase_rad{1})','hist',[],10,false,true,'linewidth',2,'color','r');  title(titles{1}); set(gca,'fontsize', 14);
% subplot(4,2,2)
% [a, phi(2),zm(2)] = circ_plot(cell2mat(Phase_rad{2})','hist',[],10,false,true,'linewidth',2,'color','r');  title(titles{2}); set(gca,'fontsize', 14);
% subplot(4,2,3)
% [a, phi(3),zm(3)] = circ_plot(cell2mat(Phase_rad{3})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{3}); set(gca,'fontsize', 14);
% subplot(4,2,4)
% [a, phi(4),zm(4)] = circ_plot(cell2mat(Phase_rad{4})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{4});set(gca,'fontsize', 14); 
% subplot(4,2,5)
% [a, phi(5),zm(5)] = circ_plot(cell2mat(Phase_rad{5})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{5}); set(gca,'fontsize', 14);
% subplot(4,2,6)
% [a, phi(6),zm(6)] = circ_plot(cell2mat(Phase_rad{6})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{6});set(gca,'fontsize', 14); 
% subplot(4,2,7)
% [a, phi(7),zm(7)] = circ_plot(cell2mat(Phase_rad{7})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{7}); set(gca,'fontsize', 14);
% subplot(4,2,8)
% [a, phi(8),zm(8)] = circ_plot(cell2mat(Phase_rad{8})','hist',[],10,false,true,'linewidth',2,'color','r'); title(titles{8});set(gca,'fontsize', 14); 
% 


%% calculating mean resultant vector length and Rayleigh test
for ii = 1:length(Phase_deg_sum_rad)
    r(ii) = circ_r(Phase_deg_sum_rad{ii}');
    p_alpha(ii) = circ_rtest(Phase_deg_sum_rad{ii}')
end


% calculate the vector length for each participant, to make anova test
R_group=[];
for ii = 1:length(Phase_allStimSubject)
    for jj = 1:length(Phase_allStimSubject{ii})
       R_group{ii}(jj) = circ_r(deg2rad(Phase_allStimSubject{ii}{jj}'));
    end
end
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
% 1,5,2,6,3,7,4,8 - resting state

% means across participants, per condition
figure; 
PLV1 = cell2mat(PLV{1}'); mPLV1 = mean(PLV1,1);         
plot(frequencies, mPLV1,'Color',COLORS(3,:),'LineWidth',2.5);

PLV2 = cell2mat(PLV{2}'); mPLV2 = mean(PLV2,1);
hold on; plot(frequencies, mPLV2,'Color',COLORS(10,:),'LineWidth',2.5);

PLV3 = cell2mat(PLV{3}'); mPLV3 = mean(PLV3,1);
hold on; plot(frequencies, mPLV3,'Color',COLORS(4,:),'LineWidth',2.5);

PLV4 = cell2mat(PLV{4}'); mPLV4 = mean(PLV4,1);
hold on; plot(frequencies,mPLV4,'Color',COLORS(11,:),'LineWidth',2.5);

PLV5 = cell2mat(PLV{5}'); mPLV5 = mean(PLV5,1);
hold on; plot(frequencies, mPLV5,'Color',COLORS(6,:),'LineWidth',2.5);

PLV6 = cell2mat(PLV{6}'); mPLV6 = mean(PLV6,1);
hold on; plot(frequencies, mPLV6,'Color',COLORS(13,:),'LineWidth',2.5);

PLV7 = cell2mat(PLV{7}'); mPLV7 = mean(PLV7,1);
hold on; plot(frequencies, mPLV7,'Color',COLORS(7,:),'LineWidth',2.5);

PLV8 = cell2mat(PLV{8}'); mPLV8 = mean(PLV8,1);
hold on; plot(frequencies, mPLV8,'Color',COLORS(14,:),'LineWidth',2.5);

% adding long resting states
figure
PLV9 = cell2mat(PLV{9}'); mPLV9 = mean(PLV9,1);
hold on; plot(frequencies, mPLV9,'Color','k','LineWidth',2.5);
PLV10 = cell2mat(PLV{10}'); mPLV10 = mean(PLV10,1);
hold on; plot(frequencies, mPLV10,'Color','k','LineWidth',2.5);



xlabel('Frequency (Hz)');
ylabel('Coherence (AU)');
%xlim([1 2.5]);
%ylim([0.1 0.3]);
 
legend('TD ISO','ASD ISO','TD Jitter S','ASD Jitter S','TD Rand','ASD Rand');

title('Phase locking values for all conditions')

set(gca,'fontsize', 14);

% For long resting state (before the experiment): 
figure; 
PLV9 = cell2mat(PLV{9}'); mPLV1 = mean(PLV9,1);         
plot(frequencies, mPLV1,'Color',COLORS(3,:),'LineWidth',2.5);
PLV10 = cell2mat(PLV{10}'); mPLV2 = mean(PLV10,1);
hold on; plot(frequencies, mPLV2,'Color',COLORS(10,:),'LineWidth',2.5);


%% %%%%%%%%% normalizing rest plv to the total rest (long) plv
mPLV1 = mPLV1./mPLV9;
mPLV2 = mPLV2./mPLV9;
mPLV3 = mPLV3./mPLV9;
mPLV4 = mPLV4./mPLV9;
mPLV5 = mPLV5./mPLV10;
mPLV6 = mPLV6./mPLV10;
mPLV7 = mPLV7./mPLV10;
mPLV8 = mPLV8./mPLV10;



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


% plot differences 
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

%% plot resting state conditions
% 1,5,2,6,3,7,4,8 - resting state

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


