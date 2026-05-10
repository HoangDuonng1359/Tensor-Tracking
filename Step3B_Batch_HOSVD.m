%% =========================================================================
% STEP 3B: BATCH HOSVD (HIGHER-ORDER SINGULAR VALUE DECOMPOSITION)
% Description: Applies global HOSVD to extract spatial basis matrices and 
% tracks projection errors to find ERN Change Points.
% =========================================================================
clear; clc; close all;

% --- 1. CONFIGURATION ---
DATA_DIR   = 'D:\Data\tensor_data\';
TENSOR_FILE = fullfile(DATA_DIR, 'Group_Dynamic_Tensor.mat');
RESULTS_DIR = fullfile(DATA_DIR, 'hosvd_results'); % Thư mục lưu kết quả riêng

fprintf('======================================================\n');
fprintf('   STARTING BATCH HOSVD ALGORITHM\n');
fprintf('======================================================\n\n');

if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR);
end

% Load the 4D Group Tensor
fprintf('[INFO] Loading Group Dynamic Tensor...\n');
load(TENSOR_FILE, 'GROUP_TENSOR');
[n_chan, ~, num_subjects, n_points] = size(GROUP_TENSOR);
fprintf('-> Tensor Loaded successfully: [%d x %d x %d x %d]\n\n', n_chan, n_chan, num_subjects, n_points);

% --- 2. ALGORITHM PARAMETERS ---
RANK_SPACE = 5;      % Spatial subspace rank (Low-rank dimension)
BLIND_GAP = 50;      % Ignore initial frames to stabilize the threshold

Subject_Results_HOSVD = struct('Subject_ID', {}, 'Change_Points', {}, 'Error_Track', {}, 'Valid', {});

%% --- 3. BATCH HOSVD PROCESSING LOOP ---
fprintf('[INFO] Executing HOSVD decomposition per subject...\n');

for s = 1:num_subjects
    subj_id = sprintf('sub-%03d', s);
    subj_tensor = squeeze(GROUP_TENSOR(:,:,s,:));
    
    if sum(subj_tensor(:)) == 0
        fprintf('   -> [%s] SKIPPED (Empty Data)\n', subj_id);
        Subject_Results_HOSVD(s).Valid = false;
        continue;
    end
    
    fprintf('   -> [%s] Processing... ', subj_id);
    
    % -- Phase 1: Tensor Unfolding & HOSVD Spatial Factors --
    % Unfold Mode-1 (Channel 1 x [Channel 2 * Time])
    H1 = reshape(subj_tensor, n_chan, n_chan * n_points);
    [U1, ~, ~] = svd(H1, 'econ');
    U1_trunc = U1(:, 1:RANK_SPACE); % Spatial Basis for Mode 1
    
    % Unfold Mode-2 (Channel 2 x [Channel 1 * Time])
    H2 = reshape(permute(subj_tensor, [2 1 3]), n_chan, n_chan * n_points);
    [U2, ~, ~] = svd(H2, 'econ');
    U2_trunc = U2(:, 1:RANK_SPACE); % Spatial Basis for Mode 2
    
    % -- Phase 2: Global Low-Rank Projection & Error Tracking --
    Error_Track = zeros(1, n_points);
    Change_Points = [];
    
    % Calculate dynamic threshold based on the first 200 frames (similar to HO-RLSL)
    temp_errors = zeros(1, 200);
    for t = 1:200
        H_t = subj_tensor(:, :, t);
        % Low-rank reconstruction using HOSVD spatial factors
        L_t = U1_trunc * (U1_trunc' * H_t * U2_trunc) * U2_trunc';
        % Sparse / Residual component
        V_t = H_t - L_t;
        temp_errors(t) = norm(V_t, 'fro');
    end
    DYNAMIC_THRESHOLD = mean(temp_errors) + 2 * std(temp_errors);
    
    % Track errors across the entire timeline
    for t = 1:n_points
        H_t = subj_tensor(:, :, t);
        
        % HOSVD Low-Rank Projection: L = U1 * S * U2'
        L_t = U1_trunc * (U1_trunc' * H_t * U2_trunc) * U2_trunc';
        V_t = H_t - L_t;
        
        residual_error = norm(V_t, 'fro');
        Error_Track(t) = residual_error;
        
        % Detect anomalies (Change Points)
        if (t > BLIND_GAP) && (residual_error > DYNAMIC_THRESHOLD)
            Change_Points = [Change_Points, t];
        end
    end
    
    % Filter consecutive Change Points (keep 1 peak per 30 frames window)
    if ~isempty(Change_Points)
        Change_Points = Change_Points([1, find(diff(Change_Points) > 30) + 1]);
    end
    
    % Store extracted features
    Subject_Results_HOSVD(s).Subject_ID = subj_id;
    Subject_Results_HOSVD(s).Change_Points = Change_Points;
    Subject_Results_HOSVD(s).Error_Track = Error_Track;
    Subject_Results_HOSVD(s).Valid = true;
    
    fprintf('Found %d ERN points.\n', length(Change_Points));
end

%% --- 4. SAVE EXTRACTED FEATURES ---
save_file = fullfile(RESULTS_DIR, 'HOSVD_Extracted_Features.mat');
save(save_file, 'Subject_Results_HOSVD');
fprintf('\n[SUCCESS] Feature extraction completed and saved to: %s\n', save_file);

%% --- 5. VISUALIZATION: HOSVD RASTER PLOT ---
fprintf('[INFO] Generating HOSVD Group-Level Raster Plot...\n');
figure('Name', 'HOSVD Group-Level ERN Change Points', 'Position', [150, 150, 1000, 500], 'Color', 'w');
hold on;

valid_count = 0;
for s = 1:num_subjects
    if Subject_Results_HOSVD(s).Valid
        valid_count = valid_count + 1;
        cps = Subject_Results_HOSVD(s).Change_Points;
        if ~isempty(cps)
            scatter(cps, repmat(valid_count, 1, length(cps)), 30, 'b', 'filled', 'MarkerEdgeColor', 'k');
        end
    end
end

title('Group-Level ERN Synchronization (HOSVD Baseline)', 'FontSize', 15, 'FontWeight', 'bold');
xlabel('Time (Frames)', 'FontSize', 12);
ylabel('Valid Subjects', 'FontSize', 12);
ylim([0, valid_count + 1]);
xlim([0, n_points]);
grid on; set(gca, 'GridAlpha', 0.3);

annotation('textbox', [0.15, 0.8, 0.3, 0.1], 'String', ...
    sprintf('Algorithm: HOSVD\nBlue dots represent detected ERN shifts'), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', 'k');

fprintf('======================================================\n');
fprintf('   PIPELINE STEP 3B FULLY COMPLETED!\n');
fprintf('======================================================\n');