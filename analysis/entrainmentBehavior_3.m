%PARAMS = paramsPreProcessingVAMP;
clearvars -except colors

originalFs = 512;
prompt = 'modality (1-V, 2-A, 3-AV)? ';  
mod = input(prompt);

prompt = 'At work (1) or home (2) ?';
loc = input(prompt);

if mod == 1 && loc == 1%visual
    dataPath = 'C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\entrainmentRawData\Visual';
    addpath('C:\Users\sbeker\Dropbox (EinsteinMed)\A ENTRAINMENT AND OSCILLATION IN ASD\Analysis');
    addTrig = 30;
elseif mod == 2 && loc == 2 % auditory
    dataPath = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Auditory';
    addpath('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    addTrig = 0;
elseif mod == 1 && loc == 2 %visual at home
     dataPath = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Visual';
    addpath('/Users/shlomit/Dropbox (EinsteinMed)/GENERAL ANALYSIS FILES');
    
end

numBlocks = 0;
moreBlocks = 1; %Add blocks of trials into the same preprocessing pipeline

%data = datapathEntrainment
%data = '/Users/shlomit/Dropbox (EinsteinMed)/A ENTRAINMENT AND OSCILLATION IN ASD/entrainmentRawData/Visual';
cd(dataPath)
subjectFile = dir(dataPath);

%%

cd(dataPath);
subNum = 0;
for i = 3:length(subjectFile) - 1   % 1:length(data)
    subNum = subNum+1;
    currentSub = subjectFile(i).name;
    %name(subNum) = sscanf(currentSub, '%d_%d');% subject ID. 11-ASD. all others - TD. 
    name{subNum} = currentSub;
    %name(i) = str2num(currentSub); 
    currentPath = [dataPath,'/',currentSub];
    cd(currentPath);
    bdfFiles = dir('*.bdf');
    if isempty(bdfFiles) ~=0
        continue
    end
    numfiles = length(bdfFiles);
    %mydata = cell(1, numfiles);
    for j = 1:numfiles
        numBlocks = numBlocks+1;

        event = ft_read_event(bdfFiles(j).name);
        event = event(:,3:end);
        if isempty(event) ~=1
        valueInd = [];
            for k = 1:size(event,2) 
                 if isempty(event(k).value) ==0 
                    valueInd(k,1) = event(k).value'; 
                 end
            end
        VALS{subNum}{j,:} = valueInd;
        triggers = [[12:23]+addTrig,110]; %33 is a bug (in presentation?). should be 43. CHECK BEFORE NEXT RUN!
        response = 100;
       %if empty(find(valueInd == 100)) 
        hit = 0;
        miss = 0;
        falseAlarm = 0;
        hitLocs = [];
        missLocs = [];
        faLocs = [];
        RThit = [];
        RTcond = [];
        %missRT = [];
        faRT = [];
        for m = 1:length(nonzeros(valueInd))-2
            trg1 = valueInd(m);
            trg2 = valueInd(m+1);
            trg3 = valueInd(m+2);
            if  ismember(trg1,triggers) && (trg2 == response) && m > 50 % 80 is when the udtr ends
          % if  ismember(trg1,triggers) && (trg2 == response || trg3 == response) && m > 80 % 80 is when the udtr ends

                hitLocs(hit+1) = m;
                hit = hit + 1;
                if trg2 == response
                    RThit(hit) = (event(m+1).sample-event(m).sample)./originalFs;
                elseif trg3 == response
                    RThit(hit) = (event(m+2).sample-event(m).sample)./originalFs;
                end
                    RTcond(hit) = valueInd(m);
            %elseif ismember(trg1,triggers) && (trg2 ~= response || trg3 ~=response) && m > 50 % 80 is when the udtr ends
            elseif ismember(trg1,triggers) && (trg2 ~= response) && m > 50 % 80 is when the udtr ends
 
                missLocs(miss+1) = m;
                miss = miss + 1;
                missCond(miss+1)  = trg1;
                %missRT(miss) = (event(m+1).sample-event(m).sample)./originalFs;
            elseif ismember(trg1,triggers)==0 && (trg2 == response || trg3 == response) && trg1 ~=100 && m > 50 % 80 is when the udtr ends
                falseAlarm = falseAlarm+1;
                faCond(subNum,falseAlarm +1) = trg1;
                faLocs(subNum,falseAlarm +1) = m;
                FA(subNum) = falseAlarm;
                %faRT(falseAlarm) = (event(m+1).sample-event(m).sample)./originalFs;          
            end
        
        end
        Behav.numbers{subNum}(j,:)= cat(1,hit,miss,falseAlarm);
        Behav.rt{subNum}{:,j} = RThit;
        Behav.hitCond{subNum}{j} = RTcond;
        Behav.miss{subNum}{j} = missLocs;
        Behav.missCond{subNum}{j} = missCond;
        Behav.FACond{subNum}{j} = faCond;
      end
    end
end

%%
numTrig = [];
for i = 1:length(VALS)
    VALS_subject{i} = cell2mat(VALS{i});
    for j = 1:length(triggers)
        numTrig = numTrig + 1; 
        VALS_trigs{i}(j) = sum(VALS_subject{i} == triggers(j));
    end
end


%% get rid of empty cells - of ones that didnt respond. 
%in the meantime - get rid of %15. 
% reaction time histograms
clear HIT COND C MEAN RT

for i = 1:length(Behav.numbers)
    HIT{i,:} = cell2mat(Behav.rt{i});
    HIT_COND{i,:} = cell2mat(Behav.hitCond{i});
    
    [C{i},ia,ic] = unique(HIT_COND{i,:}); %C is the unique trigger codes per each subjects
    for j = 1:length(C{i}-1)
        RT{i}{:,j} = HIT{i}(find(ic ==j)); % all RT, not averaged
        MEAN{i}(j) = mean(HIT{i}(find(ic ==j))); %Mean RT per trigger code per subjects (triggers according to C)
    end
end

empty = [];
for i = 1:length(HIT)
    if isempty(HIT{i})
        empty = cat(1,empty,i);
    end
end

HIT(empty) = [];
C(empty) = [];
MEAN(empty) = [];
name(empty) = [];

id_ASD = []; 
id_TD = [];
for i = 1:length(name)
    if str2num(name{i}(2))==1 || str2num(name{i}(2))==8
        id_ASD = cat(1,id_ASD,i);
    else
        id_TD = cat(1,id_TD,i);
    end
end


%% for each group seperately

% TD

Behav_TD.numbers = Behav.numbers(id_TD);
Behav_TD.rt = Behav.rt(id_TD);
Behav_TD.hitCond = Behav.hitCond(id_TD);
Behav_TD.miss = Behav.miss(id_TD);
Behav_TD.missCond = Behav.missCond(id_TD);
Behav_TD.FACond = Behav.FACond(id_TD);

MEAN_TD = MEAN(id_TD)';
names_TD = name(id_TD)';
C_TD = C(id_TD)';

allMEAN_TD(1,:) = cell2mat(C(id_TD));
allMEAN_TD(2,:) = cell2mat(MEAN(id_TD));

% miss/hit percentage - Td

%miss
missConds_TD = [];
for i = 1:length(Behav_TD.missCond)
    missConds_TD = cat(2,missConds_TD, Behav_TD.missCond{i}{1});
end
missConds_TD(missConds_TD==0) = [];
[MissCondUniq,ia,ic] = unique(missConds_TD);
 for i = 1:length(MissCondUniq)
     Miss_By_Cond_TD(i) = length(find(ic == i));
 end

 %hit
hitConds_TD = [];
for i = 1:length(Behav_TD.missCond)
    hitConds_TD = cat(2,hitConds_TD, Behav_TD.hitCond{i}{1});
end
hitConds_TD(hitConds_TD==0) = [];
[HitCondUniq,ia,ic] = unique(hitConds_TD);
 for i = 1:length(HitCondUniq)
     Hit_By_Cond_TD(i) = length(find(ic == i));
 end

%% ASD
Behav_ASD.numbers = Behav.numbers(id_ASD);
Behav_ASD.rt = Behav.rt(id_ASD);
Behav_ASD.hitCond = Behav.hitCond(id_ASD);
Behav_ASD.miss = Behav.miss(id_ASD);
Behav_ASD.missCond = Behav.missCond(id_ASD);
Behav_ASD.FACond = Behav.FACond(id_ASD);

MEAN_ASD = MEAN(id_ASD)';
names_ASD = name(id_ASD)';
C_ASD = C(id_ASD)';

allMEAN_ASD(1,:) = cell2mat(C(id_ASD));
allMEAN_ASD(2,:) = cell2mat(MEAN(id_ASD));

% miss/hit percentage - ASD

% miss
missConds_ASD = [];
for i = 1:length(Behav_ASD.missCond)
    missConds_ASD = cat(2,missConds_ASD, Behav_ASD.missCond{i}{1});
end
missConds_ASD(missConds_ASD==0) = [];
[MissCondUniq,ia,ic] = unique(missConds_ASD);
 for i = 1:length(MissCondUniq)
     Miss_By_Cond_ASD(i) = length(find(ic == i));
 end

 %hit
hitConds_ASD = [];
for i = 1:length(Behav_ASD.missCond)
    hitConds_ASD = cat(2,hitConds_ASD, Behav_ASD.hitCond{i}{1});
end
hitConds_ASD(hitConds_ASD==0) = [];
[HitCondUniq,ia,ic] = unique(hitConds_ASD);
 for i = 1:length(HitCondUniq)
     Hit_By_Cond_ASD(i) = length(find(ic == i));
 end

%% now average RT per subject and trigger code

%RTcond(RTcond==33) = 43; %again, takes care of the '33' bug

[indTD,ia,ic] = unique(allMEAN_TD(1,:) );
RT_TD = [];
for i = 1:length(indTD)
    locs = find(ic==i);
    RT_TD{i} = allMEAN_TD(2,locs);
end

[indASD,ia,ic] = unique(allMEAN_ASD(1,:) );
RT_ASD = [];
for i = 1:length(indASD)
    locs = find(ic==i);
    %RT_ASD{i} = mean(allMEAN_ASD(2,locs));
        RT_ASD{i} = allMEAN_ASD(2,locs);

end

%% plot (mick's function)
  colors = rand(13,3);

% TD
st_boxdotplot([1:length(RT_TD)],RT_TD,colors,'iqr',[],[],[],0.5,36,0.5,[],1)  %mick RT plot func
title('RT TD');
ylim([0.2 1.5]);
set(gca,'fontsize', 16); 
set(gca,'xtick',[1:length(RT_TD)]);
% ASD
st_boxdotplot([1:length(RT_ASD)],RT_ASD,colors,'iqr',[],[],[],0.5,36,0.5,[],1)  %mick RT plot func
title('RT ASD');
ylim([0.2 1.5]);
set(gca,'fontsize', 16);
set(gca,'xtick',[1:length(RT_ASD)]);

%%
figure;
h1 = histogram(RThit(RThit>0.2),40);
title('all reaction times');
xlim([0 1.8]);
xlabel('time (sec.)');
ylabel('number of trials');
   
figure;

conds = unique(RTcond);
subplots = [1,2,2,3,3,3,3,3,4,5,6,7]; %to overlay hists of triggers of the same condition on one subplot
upperlim = 0.6;
lowerlim = 0.12;
for i = 1:length(conds)
   RTperCond{i} = RThit(RTcond==conds(i));
   RTsmall{i} = length(find(RTperCond{i}(RTperCond{i}<0.2)));
   subplot(4,2,subplots(i))
   %h1 = histogram(RTperCond{i}(RTperCond{i}>0.2),35);
   F = find(RTperCond{i}>lowerlim & RTperCond{i}<upperlim);
   h1 = histogram(RTperCond{i}(F),20);
   title(num2str(subplots(i)));
   hold on;
   meanRT(i) = mean(RTperCond{i}(F));
    %h1.Normalization = 'probability';
	%h1.BinWidth = 0.02;

end

figure; bar(meanRT([1,9,11]));
ylim([0.3 0.6]);

%% Percentage of correct responses
clear percentage

for i = 1:13    % 1:length(data)
    cd(data{i});
    bdfFiles = dir('*.bdf');
    numfiles = length(bdfFiles);
    %mydata = cell(1, numfiles);
    for j = 1:length(bdfFiles)
        numBlocks = numBlocks+1;
        
        dataPath = bdfFiles(j).folder
        DATA = bdfFiles(j).name;
        SCEN = sscanf(DATA,'%s'); 
        event = ft_read_event([dataPath '/' DATA]);
        valueInd = [];
        for k = 3:size(event,2) 
            if isempty(event(k).value) ==0 
                valueInd(k) = event(k).value; 
            end
        end
        triggers = cfg.trialdef.eventvalue;
        correct = [7,37,67,97,127,157,187];
        incorrect = [9,39,69,129,159,189];
        
        percentTrig = [];
        
        for m = 1:length(nonzeros(valueInd))
            trg1 = valueInd(m);
            trg2 = valueInd(m+1);
            trg3 = valueInd(m+1)
            if  ismember(trg1,triggers) && ismember(trg2,correct) || ismember(trg3,correct);
                percentTrig(m) = 1; 
            elseif ismember(trg1,triggers) && ismember(trg2,incorrect) || ismember(trg3,incorrect);
                percentTrig(m) = 2;
            end
        end
        
        percentage{i}(j,1) = nnz(percentTrig==1)/(nnz(percentTrig==1)+nnz(percentTrig==2));
        if i == 1
            percentage{i}(j,2) = str2num(SCEN(9));
        else
            percentage{i}(j,2) = str2num(SCEN(11));
        end
    end
    %MEANRT_1{i} = real(meanRT);
end


uisave('percentage','percentage');


    %%
    % add indices of conditions to the result matrix
    
   MEANRT_1 = percentage;

    MEANRT_stat = [];
    indices = [];
    for i = 1:length(MEANRT_1)
        MEANRT_stat = cat(1,MEANRT_stat,MEANRT_1{i});
        indices = cat(1,indices, ones(length(MEANRT_1{i}),1)*i);
        
    end
    MEANRT_stat(:,3) = indices;
    %
%     
%     for j = 1:length(MEANRT_1)
%         subInd = find(MEANRT_stat(:,3)==j);
%         cond = unique(MEANRT_stat(subInd,2));
%         for k = 1:length(cond)
%             condIndSub = find(MEANRT_stat(subInd,3)==k);
%             condMean(j,k) = mean(MEANRT_stat(MEANRT_stat(condIndSub,2)== cond(k)));
%             
%         end
%     end
%     
    
    %%
    %condMean = NaN(11,6);

    clear condMean
    
    for j = 1:length(MEANRT_1)
        subInd = find(MEANRT_stat(:,3)==j);
        cond = unique(MEANRT_stat(subInd,2));
        for k = 1:length(cond)
            condIndSub = find(MEANRT_stat(:,3)==j & MEANRT_stat(:,2)==cond(k));
            
            condMean(j,k) = mean(MEANRT_stat(condIndSub,1));
            
        end
    end
    
    [p,tbl,stats] = anova1(MEANRT_stat(:,1),MEANRT_stat(:,2));
    
    
    %% arrange and export to excel
    xlswrite('mean_RT',MEANRT_stat,condMean);
    
    %% import means from excel
%     cond1 = xlsread('means .xlsx',1,'C2:C104');
%     cond7 = xlsread('means .xlsx',1,'F2:F81');
% figure;
% h1 = histogram(cond1,20);
% title('isochronous');
% xlim([0 1.8]);
% xlabel('time (sec.)');
% ylabel('number of trials');