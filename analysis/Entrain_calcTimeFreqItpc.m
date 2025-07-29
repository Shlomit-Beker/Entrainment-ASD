clear itcAll dataTemp spectAll spectrumEst itcGroup tfr data_bl_TFR
%% calculate TFR and ITC for each channel TAKES TIME! DO NOT RUN IF NOT INTENTIONALLY!
tic

gwidthWlt = 3;  
freqoi = 0.1:0.05:5;
widthWlt = linspace(3,8,100);
CHAN = 'POz';
chns = find(strcmp(ERP{1}{1}.label,CHAN)); % enter the ID of channels to analyze
%numTrl = 1; % enter the number of trials
t = [1:length(ERPb{1}{1}.avg)]/256;

% calculate ITC
% try one to initialize size of timeoi and freqoi
% data is a fieldtrip structure with single trial data
% inputs: 
% dataTemp - single trial data matrix (chans X time)
% t - time vector in seconds
clc
if Signal == 6
    DATAa = DATA([1,13,11,23,12,24,9,21]);
    %DATAa = [DATA(1),DATA(13),DATA(11),DATA(23),DATA(12),DATA(24),DATA(9),DATA(21)];

elseif Signal == 7
    DATAa = DATA([1,4,6,7,9,13,15,16]);
    %DATAa = [DATA(1),DATA(4),DATA(6),DATA(7),DATA(9),DATA(13),DATA(15),DATA(16)];
end

DATAr = rerefData(DATAa,37);


dataTemp = DATAr{1}{1}{1}(chns,:); 
[~,freqoi,timeoi] = ft_specest_wavelet(dataTemp, t, 'freqoi', freqoi,'width', widthWlt, 'gwidth',gwidthWlt);


%% for all channels (skip to next section if only one chan is needed)
for Group =[1,2]
    clear DATA_I
    for k = 1:length(DATAr{Group})
        for l = 1:length(DATAr{Group}{k})
            DATA_I{k}(:,:,l) = DATAr{Group}{k}{l};
        end
    end
  
    %%%
    %itcAll = zeros(length(chns),length(freqoi),length(timeoi));
    %phiAll = zeros(num aTrl,length(freqoi),length(timeoi));
    clear itcAll 
    for CHAN = 1:length(ERP{1}{1}.label);
        clear itc
        for k =  1:length(DATA_I)  % loop on subjects
            % initialize matrix for ITC
            % initialize matrix for spectrogram
            %for chnI = 1 : length(chns)
            % this matrix holds the instantaneous phase for each trial, frequency
            % and time point
            numTrl = size(DATA_I{k},3);
            spectAll = zeros(numTrl,length(freqoi),length(timeoi));
            for trlI = 1 : numTrl
                % select the correct trial and channel
                %dataTemp = data.trial{trlI}(chnI,:);
                dataTemp = DATA_I{k}(CHAN,:,trlI);
                % calculate time frequency analysis using wavelets
                [spectrumEst,freqoi,timeoi] = ft_specest_wavelet(dataTemp, t, 'freqoi', freqoi, 'width', widthWlt, 'gwidth',gwidthWlt);
                spectAll(trlI,:,:) = spectrumEst(1,:,:);
            end
            % calculate itc for a single channel across trials
            itc{k} = it_calcITC(spectAll);
            % spectAll_abs = abs(spectAll);
            % spectAll_abs_mean = squeeze(mean(spectAll_abs,1));              % for TFR 
            % tfr{k}(1,:,:) = spectAll_abs_mean;
        end
        %sumItc = zeros(6,897); 
        sumItc = zeros(size(itc{1}));
        for k = 1:length(itc) %for averaging across group
            %sqItc = squeeze(itc{k});
            sumItc = sumItc+itc{k};
        end
      itcAll{CHAN} = sumItc./length(DATA{Group});

    end
    itcGroup{Group} = itcAll;
    
end

toc
%itcDiff = itcAllTd - itcAllAsd;

%% for specific channels
clear itcAll dataTemp spectAll spectrumEst itcGroup tfr itc

Group = 2;

clear DATA_I
for cond = 1:length(DATAr)
    for k = 1:length(DATAr{cond})
        for l = 1:length(DATAr{cond}{k})
            DATA_I{cond}{k}(:,:,l) = DATAr{cond}{k}{l};
        end
    end
end

if Group == 1

    DATA_G = DATA_I(1:4);
    group = 'TD';
else
    DATA_G = DATA_I(5:8);
    group = 'ASD'
end

    %itcAll = zeros(length(chns),length(freqoi),length(timeoi));
    %phiAll = zeros(num aTrl,length(freqoi),length(timeoi));
    for Cond = 1:length(DATA_G)
    %clear itcAll 
    %for C = 1:length(ERP{1}{1}.label);
        %clear itc
        for k =  1:length(DATA_G{Cond})  % loop on subjects
            % initialize matrix for ITC
            % initialize matrix for spectrogram
            %for chnI = 1 : length(chns)
            % this matrix holds the instantaneous phase for each trial, frequency
            % and time point
            numTrl = size(DATA_G{Cond}{k},3);
            spectAll = zeros(numTrl,length(freqoi),length(timeoi));
            for trlI = 1 : numTrl
                % select the correct trial and channel
                %dataTemp = data.trial{trlI}(chnI,:);
                dataTemp = DATA_G{Cond}{k}(chns,:,trlI);
                % calculate time frequency analysis using wavelets
                [spectrumEst,freqoi,timeoi] = ft_specest_wavelet(dataTemp, t, 'freqoi', freqoi, 'width', widthWlt, 'gwidth',gwidthWlt);
                spectAll(trlI,:,:) = spectrumEst(1,:,:);
            end
            % calculate itc for a single channel across trials
            itc{Cond}{k} = it_calcITC(spectAll);
            spectAll_abs = abs(spectAll);
            spectAll_abs_mean = squeeze(mean(spectAll_abs,1));              % for TFR 
            tfr{Cond}{k}(1,:,:) = spectAll_abs_mean;
        end
        %sumItc = zeros(6,897); 
        sumItc = zeros(size(itc{Cond}{1}));
        for k = 1:length(itc{Cond}) %for averaging across group
            %sqItc = squeeze(itc{k});
            sumItc = sumItc+itc{Cond}{k};
        end
      itcAll{Cond} = sumItc./length(DATA_G{Cond});

    end
    itcGroup{Cond} = itcAll;

 %% TFR

sumtfr = zeros(length(freqoi),length(timeoi));
        nans = 0;
     for g = 1:length(tfr)
        for k = 1:length(tfr{g}) %for averaging across cond
             sqtfr = squeeze(tfr{g}{k});
%              if sum(sum(isnan(sqtfr))) == size(sqtfr,1)*size(sqtfr,2)
%                  sqtfr = [];
%                  nans = nans+1;
%                  k
%                  continue
%              end
             sumtfr = sumtfr+sqtfr;
        end

     dataTFR{g} = sumtfr./length(tfr{g});
     end

% Baselining the TFR by subtracting the TFR from the pre-stim period, at the average level (as in Hu and Iannetti, 2014). 
% done per one group each time on "data". 
% and PLOT TFR and ITPC


for jj = 1:length(dataTFR)

    timeBefore = [3000:3400]; %time before the first Epoch; DO IT PER EACH FREQUENCY!

    normWindow = nanmean(dataTFR{jj}(:,timeBefore),2);

    for i = 1:length(timeoi)
        data_bl_TFR{jj}(:,i) = dataTFR{jj}(:,i)./normWindow;
        %data_bl_TFR{jj}(:,i) = dataTFR{jj}(:,i);
    end


    %% Plot TFR
    %Data = data_bl_TFR{jj};
  
    Data = db(data_bl_TFR{jj});

    N = 500;
    timeReduction = [1:length(timeoi)]; %change according to the NaNs in the data matrix. Try to get rid of as many as possible.
    Timeoi = timeoi(timeReduction);
    [n, m] = size(Data);
    [x,y] = meshgrid(Timeoi,freqoi); % low-res grid
    [x2,y2] = meshgrid(Timeoi(1):1/N/5:Timeoi(end),freqoi(1):.01:freqoi(end));  %high-res grid
    dataInterp = interp2(x,y,Data, x2,y2, 'linear'); %interpolate up

%     figure;
%     subplot(3,3,[1,2,3,4,5,6])
    %f = surf(x2,y2,dataInterp);
%     f.EdgeColor = 'none';
%     f.FaceColor = 'interp';
%     f.FaceLighting = 'gouraud';
%     set(gca,'ydir','normal')
%     ylabel('Frequency (Hz)')
%     xlabel('Time (Sec.)');
%     colorbar;
%     colormap viridis;
%     ax = gca;

    %caxis(ax.CLim)
    %caxis([-0.47 1.05]);
    %ax.CLim = [-0.47 1.05];

%     view(0,90)
%     axis tight
%     set(gca,'FontSize',16) % Creates an axes and sets its FontSize to 18
    %title(['TFR map (Pz): ',num2str(jj)])    
    %subplot(3,3,[7,8,9])
    %plot(Timeoi,mean(Data([12:18],:)),'k','LineWidth',4)
%     figure(Group)
%     plot(Timeoi,mean(Data([12:18],:)),'LineWidth',3)
%     title([group, ' TFR 1.5Hz (Pz)'])    
%     hold on

    %% Plot ITC
   
    %Data = itcAll{jj};
    Data = itcAll{1}-itcAll{2};

    N = 500;
    timeReduction = [1:length(timeoi)]; %change according to the NaNs in the data matrix. Try to get rid of as many as possible.
    Timeoi = timeoi(timeReduction);
    [n, m] = size(Data);
    [x,y] = meshgrid(Timeoi,freqoi); % low-res grid
    [x2,y2] = meshgrid(Timeoi(1):1/N/5:Timeoi(end),freqoi(1):.01:freqoi(end));  %high-res grid
    dataInterp = interp2(x,y,Data, x2,y2, 'linear'); %interpolate up


%     figure(Group+2)
%     plot(Timeoi,mean(Data([12:18],:)),'LineWidth',3)
%     hold on
%     title([group, '  ITPC 1.5Hz (Pz): '])
    
end


%% Plot



    figure;
    subplot(3,3,[1,2,3,4,5,6])
    f = surf(x2,y2,dataInterp);
    f.EdgeColor = 'none';
    f.FaceColor = 'interp';
    f.FaceLighting = 'gouraud';
    set(gca,'ydir','normal')
    ylabel('Frequency (Hz)')
    xlabel('Time (Sec.)');
    colorbar;
    colormap inferno;
    ax = gca;

    ax.CLim = [-0.1 0.4];

    view(0,90)
    axis tight
    set(gca,'FontSize',16) % Creates an axes and sets its FontSize to 18
    ylim([0.5 5])
    xlim([1 10])
    %title(['ITPC map (Pz): ',num2str(jj)])
    %subplot(3,3,[7,8,9])
    %plot(Timeoi,mean(Data([12:18],:)),'r','LineWidth',4)

     xline(5,'w', 'lineWidth',2)
%     yline(1,'--w', 'lineWidth',2)
%     yline(2,'--w', 'lineWidth',2)

    subplot(3,3,[7,8,9])

Diff_ISO_Rand = nanmean(itcAll{1}([12:18],:))-nanmean(itcAll{2}([12:18],:));
plot(timeoi, nanmean(itcAll{1}([12:18],:))); 
hold on; 
plot(timeoi, nanmean(itcAll{2}([12:18],:))); 
hold on; plot(timeoi, sgolayfilt(Diff_ISO_Rand,3,51),'k','LineWidth',2);
ylim([-0.2 0.7])
title([group, '  ITPC Iso-Rand diff (Pz): ']);


