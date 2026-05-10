%% =========================================================================
% STEP 3: BATCH HO-RLSL (HIGHER-ORDER ROBUST LOCAL SUBSPACE TRACKING)
% Description: Applies Low-rank & Sparse decomposition across all subjects.
% Extracts ERN Change Points for future FCCA Clustering.
% =========================================================================
clear; clc; close all;

% --- 1. CONFIGURATION ---
DATA_DIR    = 'D:\Data\tensor_data\';
TENSOR_FILE = fullfile(DATA_DIR, 'Group_Dynamic_Tensor.mat');
RESULTS_DIR = fullfile(DATA_DIR, 'ho_rlsl_results');

fprintf('======================================================\n');
fprintf('   STARTING BATCH HO-RLSL TRACKING ALGORITHM\n');
fprintf('======================================================\n\n');

if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR);
end

fprintf('[INFO] Loading Group Dynamic Tensor...\n');
load(TENSOR_FILE, 'GROUP_TENSOR');
[n_chan, ~, num_subjects, n_points] = size(GROUP_TENSOR);
fprintf('-> Tensor Loaded successfully: [%d x %d x %d x %d]\n\n', n_chan, n_chan, num_subjects, n_points);

%% --- 2. ALGORITHM PARAMETERS (TUNED) ---
T_TRAIN   = 50;     % Baseline frames for subspace initialization
LAMBDA    = 0.15;   % Soft-thresholding penalty for sparse recovery
RANK_R    = 5;      % Subspace rank dimension
BLIND_GAP = 20;     % Ignored frames post-training to prevent transient errors

Subject_Results = struct('Subject_ID', {}, 'Change_Points', {}, 'Error_Track', {}, 'Valid', {});

%% --- 3. BATCH TRACKING LOOP ---
fprintf('[INFO] Executing HO-RLSL decomposition per subject...\n');

for s = 1:num_subjects
    subj_id = sprintf('sub-%03d', s);
    subj_tensor = squeeze(GROUP_TENSOR(:,:,s,:));
    
    % Safeguard for missing or empty subject data
    if sum(subj_tensor(:)) == 0
        fprintf('   -> [%s] SKIPPED (Empty Data)\n', subj_id);
        Subject_Results(s).Valid = false;
        continue;
    end
    
    fprintf('   -> [%s] Processing... ', subj_id);
    
    % -- Phase 1: Subspace Initialization --
    H_train = mean(subj_tensor(:, :, 1:T_TRAIN), 3);
    [U, ~, ~] = svd(H_train);
    P = U(:, 1:RANK_R);
    
    Error_Track = zeros(1, n_points);
    Change_Points = [];
    
    % Pre-run to establish a subject-specific dynamic threshold
    temp_errors = zeros(1, 200);
    for t = (T_TRAIN + 1):(T_TRAIN + 200)
        H_t = subj_tensor(:, :, t);
        Phi = eye(n_chan) - P * P';
        Q_t = Phi * H_t;
        V_t = sign(Q_t) .* max(abs(Q_t) - LAMBDA, 0);
        temp_errors(t - T_TRAIN) = norm(Q_t - Phi * V_t, 'fro');
    end
    
    % Threshold = mean + 2 standard deviations of initial projection errors
    DYNAMIC_THRESHOLD = mean(temp_errors) + 2 * std(temp_errors);
    
    % -- Phase 2: Recursive Tracking --
    for t = (T_TRAIN + 1):n_points
        H_t = subj_tensor(:, :, t);
        
        % Projection and sparse recovery
        Phi = eye(n_chan) - P * P';
        Q_t = Phi * H_t;
        V_t = sign(Q_t) .* max(abs(Q_t) - LAMBDA, 0);
        L_t = H_t - V_t;
        
        residual_error = norm(Q_t - Phi * V_t, 'fro');
        Error_Track(t) = residual_error;
        
        % Change Point detection (ignoring the blind gap)
        if (t > T_TRAIN + BLIND_GAP) && (residual_error > DYNAMIC_THRESHOLD) 
            Change_Points = [Change_Points, t];
            
            % Adaptive subspace update
            [U_new, ~, ~] = svd(L_t);
            P = U_new(:, 1:RANK_R);
        end
    end
    
    % Filter consecutive points (min 30-frame distance)
    if ~isempty(Change_Points)
        Change_Points = Change_Points([1, find(diff(Change_Points) > 30) + 1]);
    end
    
    % Store extracted features
    Subject_Results(s).Subject_ID = subj_id;
    Subject_Results(s).Change_Points = Change_Points;
    Subject_Results(s).Error_Track = Error_Track;
    Subject_Results(s).Valid = true;
    
    fprintf('Found %d ERN points.\n', length(Change_Points));
end

%% --- 4. SAVE EXTRACTED FEATURES ---
save_file = fullfile(RESULTS_DIR, 'HO_RLSL_Extracted_Features.mat');
save(save_file, 'Subject_Results');
fprintf('\n[SUCCESS] Feature extraction saved to: %s\n', save_file);

%% --- 5. VISUALIZATION: GROUP-LEVEL RASTER PLOT ---
fprintf('[INFO] Generating Group-Level ERN Raster Plot...\n');
figure('Name', 'Group-Level ERN Change Points', 'Position', [100, 100, 1000, 500], 'Color', 'w');
hold on;

valid_count = 0;
for s = 1:num_subjects
    if Subject_Results(s).Valid
        valid_count = valid_count + 1;
        cps = Subject_Results(s).Change_Points;
        
        if ~isempty(cps)
            scatter(cps, repmat(valid_count, 1, length(cps)), 30, 'r', 'filled', 'MarkerEdgeColor', 'k');
        end
    end
end

title('Group-Level ERN Synchronization (HO-RLSL)', 'FontSize', 15, 'FontWeight', 'bold');
xlabel('Time (Frames)', 'FontSize', 12);
ylabel('Valid Subjects', 'FontSize', 12);
ylim([0, valid_count + 1]);
xlim([0, n_points]);
grid on; set(gca, 'GridAlpha', 0.3);

annotation('textbox', [0.15, 0.8, 0.3, 0.1], 'String', ...
    sprintf('Total Valid Cohort: %d Subjects\nRed dots represent detected ERN shifts', valid_count), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', 'k');

fprintf('======================================================\n');
fprintf('   PIPELINE STEP 3 FULLY COMPLETED!\n');
fprintf('======================================================\n');