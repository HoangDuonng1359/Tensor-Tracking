%% =========================================================================
% STEP 3C: ALGORITHM BENCHMARKING (HO-RLSL vs HOSVD)
% Description: Quantitative metrics and visualization (No toolboxes required)
% =========================================================================
clear; clc; close all;

% --- 1. SETUP & DATA AGGREGATION ---
DATA_DIR = 'D:\Data\tensor_data\';
HORLSL_FILE = fullfile(DATA_DIR, 'ho_rlsl_results', 'HO_RLSL_Extracted_Features.mat');
HOSVD_FILE  = fullfile(DATA_DIR, 'hosvd_results', 'HOSVD_Extracted_Features.mat');

fprintf('=== FINAL BENCHMARKING: HO-RLSL vs HOSVD ===\n\n');

load(HORLSL_FILE, 'Subject_Results');
load(HOSVD_FILE, 'Subject_Results_HOSVD');

num_subjects = length(Subject_Results);
valid_idx = [Subject_Results.Valid];

horlsl_all = [];
hosvd_all = [];
horlsl_counts = zeros(1, num_subjects);
hosvd_counts = zeros(1, num_subjects);

for s = 1:num_subjects
    if valid_idx(s)
        horlsl_all = [horlsl_all, Subject_Results(s).Change_Points];
        horlsl_counts(s) = length(Subject_Results(s).Change_Points);
        
        hosvd_all = [hosvd_all, Subject_Results_HOSVD(s).Change_Points];
        hosvd_counts(s) = length(Subject_Results_HOSVD(s).Change_Points);
    end
end

h_counts = horlsl_counts(valid_idx);
s_counts = hosvd_counts(valid_idx);

% --- 2. QUANTITATIVE METRICS ---
fprintf('[QUANTITATIVE ANALYSIS]\n');
fprintf('------------------------------------------------------\n');
fprintf('%-30s | %-10s | %-10s\n', 'Metric', 'HO-RLSL', 'HOSVD');
fprintf('------------------------------------------------------\n');
fprintf('%-30s | %-10d | %-10d\n', 'Total ERN Points Detected', length(horlsl_all), length(hosvd_all));
fprintf('%-30s | %-10.2f | %-10.2f\n', 'Avg Points per Subject', mean(h_counts), mean(s_counts));
fprintf('%-30s | %-10.2f | %-10.2f\n', 'Temporal Std Dev (Frames)', std(horlsl_all), std(hosvd_all));
fprintf('------------------------------------------------------\n\n');

if std(horlsl_all) < std(hosvd_all)
    fprintf('-> CONCLUSION: HO-RLSL exhibits higher temporal concentration (lower Std Dev).\n\n');
else
    fprintf('-> CONCLUSION: HOSVD exhibits higher temporal concentration.\n\n');
end

% --- 3. VISUALIZATION ---
fprintf('[INFO] Generating comparison visualizations...\n');
figure('Name', 'Algorithm Benchmarking', 'Position', [100 100 1100 500], 'Color', 'w');

% Subplot 1: Temporal Distribution (Histogram)
subplot(1, 2, 1);
edges = 0:150:2048; 
histogram(horlsl_all, edges, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5);
hold on;
histogram(hosvd_all, edges, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5);
title('Temporal Density of ERN Detection', 'FontSize', 12);
xlabel('Time (Frames)');
ylabel('Detection Count');
legend('HO-RLSL (Adaptive)', 'HOSVD (Static)');
grid on;

% Subplot 2: Sensitivity (Bar chart with error bars)
subplot(1, 2, 2);
means = [mean(h_counts), mean(s_counts)];
stds = [std(h_counts), std(s_counts)];

b = bar(means, 'FaceColor', 'flat');
b.CData(1,:) = [0.8 0.2 0.2];
b.CData(2,:) = [0.2 0.2 0.8];
hold on;

errorbar(1:2, means, stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'HO-RLSL', 'HOSVD'});
title('Sensitivity: Avg Points per Subject', 'FontSize', 12);
ylabel('Detected Points');
grid on;

fprintf('-> ALL GRAPHS GENERATED\n');