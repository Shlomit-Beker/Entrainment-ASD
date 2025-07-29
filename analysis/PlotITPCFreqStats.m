ITPCFreqMax = xlsread('ITPCFreqMax.xlsx');
COLORS = rand(8,3);

ITPC = [plotITPC(1,1);plotITPC(1,2);plotITPC(2,1);plotITPC(2,2);plotITPC(3,1);plotITPC(3,2)...
    ;plotITPC(4,1);plotITPC(4,2)];

st_boxdotplot([1:8],ITPC,COLORS,'iqr',[],[],[],0.3,45,0.5,[],1);
% groupNames = {'TD 1st','TD 4th','ASD 1st','ASD 4th'};
% set(gca,'xtick',[1:4],'xticklabel',groupNames)
% title('CNV for 1st and 4th stimuli');
% set(gca,'fontsize', 15);