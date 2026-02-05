clear; clc; close all;
addpath(genpath('functions'));
%% Processing
%BAND_NAME = ["delta", "theta", "alpha", "beta", "gamma"];
BAND_NAME = ["alpha", "beta", "gamma"];
INFO = [];
for i = 1:length(BAND_NAME)
    band_to_load = BAND_NAME(i);
    load("data/" + band_to_load + ".mat");
    MAT = zeros(32, 7);
    MAT(2:27, :) = channel_event_acc;
    INFO(i, :, :) = MAT;
end
%% Assume run after creating INFO (INFO: [3 x 32 x 7])

alpha = squeeze(INFO(1,:,:));   % 32x7 matrix
beta  = squeeze(INFO(2,:,:));
gamma = squeeze(INFO(3,:,:));

figure('Color','w');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

plotHeat(alpha, 'alpha');
plotHeat(beta , 'beta');
plotHeat(gamma, 'gamma');


%% Save
filename = sprintf('result/InfoVec.npy');
writeNPY(INFO, filename); % Save in npy format


%% -------- Local Function --------
function plotHeat(M, ttl)
    [R,C] = size(M);

    nexttile;
    imagesc(M);
    set(gca,'YDir','reverse');      % Top row is channel 1
    colormap(parula);              % Change to 'hot' etc if desired
    colorbar;
    title(ttl);

    % Axis ticks (label as needed)
    xticks(1:C); yticks(1:R);
    xlabel('Class (1~7)'); ylabel('Channel (1~32)');

    hold on;

    % (1) Cell boundary grid lines: drawing with 0.5 spacing creates cell-like appearance
    for x = 0.5 : 1 : C+0.5
        line([x x], [0.5 R+0.5], 'Color',[0.85 0.85 0.85], 'LineWidth',0.8);
    end
    for y = 0.5 : 1 : R+0.5
        line([0.5 C+0.5], [y y], 'Color',[0.85 0.85 0.85], 'LineWidth',0.8);
    end
end