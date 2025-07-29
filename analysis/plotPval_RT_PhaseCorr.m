
% Topoplots of correlations

p_values = rho{1}; % Your 64 p-values here
chan_labels = relevantChannels; % Channel names for 64 channels
% Create FieldTrip data structure
data = [];
data.label = chan_labels; % Channel names
%data.dimord = 'chan';
data.powspctrm = p_values'; % The values to plot, transpose if needed
data.freq = 1;  
% data.avg = [];            % Field for the values to be plotted
% data.avg.pow = p_values'; % Store p-values here
%data.powspctrm = -log10(data.powspctrm);

cfg = [];
cfg.layout = lay; % Electrode layout
cfg.marker = 'on'; % Show electrode positions
cfg.comment = 'no'; % Remove additional comments
%cfg.colorbar = 'yes'; % Add a colorbar
cfg.parameter = 'powspctrm';
%cfg.zlim = [0, max(data.powspctrm)]; % Set color scale range (optional)

% Normalize p-values for better visualization if required

% Plot
figure; 
ft_topoplotTFR(cfg, data);
%%
colormap jet
ax = gca;
ax.CLim = [0.3 1];


%%


% Topoplots of 1.5 oscillation amplitudes

values = circAmp_channels{8}; % Your 64 p-values here
chan_labels = relevantChannels; % Channel names for 64 channels
% Create FieldTrip data structure
data = [];
data.label = chan_labels; % Channel names
%data.dimord = 'chan';
data.powspctrm = values'; % The values to plot, transpose if needed
data.freq = 1;  
% data.avg = [];            % Field for the values to be plotted
% data.avg.pow = p_values'; % Store p-values here
%data.powspctrm = -log10(data.powspctrm);

cfg = [];
cfg.layout = lay; % Electrode layout
cfg.marker = 'on'; % Show electrode positions
cfg.comment = 'no'; % Remove additional comments
%cfg.colorbar = 'yes'; % Add a colorbar
cfg.parameter = 'powspctrm';
%cfg.zlim = [0, max(data.powspctrm)]; % Set color scale range (optional)

% Normalize p-values for better visualization if required

% Plot
figure; 
ft_topoplotTFR(cfg, data);
%%
%colormap inferno
ax = gca;
ax.CLim = [-1 1.5];




