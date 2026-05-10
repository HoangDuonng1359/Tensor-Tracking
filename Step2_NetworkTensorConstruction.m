%% =========================================================================
% BATCH PLV CALCULATION & DYNAMIC TENSOR CONSTRUCTION
% Description: Extracts phase using custom FFT-Hilbert, calculates Phase 
% Locking Value (PLV) for 40 subjects, and constructs the final Group Tensor.
% =========================================================================
clear; clc; close all;

% --- 1. CONFIGURATION ---
INPUT_DIR  = 'D:\Data\preprocessed_data\';
OUTPUT_DIR = 'D:\Data\tensor_data\';
NUM_SUBJECTS = 40;
NUM_CHANNELS = 30;

fprintf('======================================================\n');
fprintf('   STARTING NETWORK CONNECTIVITY (PLV) COMPUTATION\n');
fprintf('======================================================\n\n');

% Create output directory if it doesn't exist
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
    fprintf('[INFO] Created new output directory: %s\n', OUTPUT_DIR);
end

%% --- 2. INITIALIZE THE GRAND TENSOR ---
% Dynamically detect the number of time points from the first valid subject
temp_file = fullfile(INPUT_DIR, 'sub-001_Preprocessed.mat');
if ~exist(temp_file, 'file')
    error('Cannot find sub-001 to initialize tensor. Check INPUT_DIR.');
end
load(temp_file, 'EEG');
NUM_TIMEPOINTS = size(EEG.data, 2);

% Pre-allocate the 4D Tensor to prevent memory fragmentation
% Dimensions: [Channels(30) x Channels(30) x Subjects(40) x Time(2048)]
GROUP_TENSOR = zeros(NUM_CHANNELS, NUM_CHANNELS, NUM_SUBJECTS, NUM_TIMEPOINTS);
fprintf('[INFO] Pre-allocated Group Tensor size: %d x %d x %d x %d\n\n', ...
    NUM_CHANNELS, NUM_CHANNELS, NUM_SUBJECTS, NUM_TIMEPOINTS);

%% --- 3. BATCH PROCESSING LOOP ---
for s = 1:NUM_SUBJECTS
    subj_id = sprintf('sub-%03d', s);
    file_path = fullfile(INPUT_DIR, sprintf('%s_Preprocessed.mat', subj_id));
    
    fprintf('------------------------------------------------------\n');
    fprintf('[%s] Loading preprocessed data...\n', subj_id);
    
    if ~exist(file_path, 'file')
        fprintf('   [WARNING] Data missing for %s. Skipping.\n', subj_id);
        continue;
    end
    
    % Load the 'EEG' struct into workspace
    load(file_path, 'EEG');
    [~, n_points, n_epochs] = size(EEG.data);
    
    % --- Step A: Custom Hilbert Transform via FFT ---
    fprintf('   -> Extracting signal phase (FFT-Hilbert method)...\n');
    
    % Construct analytic signal filter
    h = zeros(n_points, 1);
    h(1) = 1;               
    h(n_points/2 + 1) = 1;  
    h(2:n_points/2) = 2;    
    
    phase_data = zeros(NUM_CHANNELS, n_points, n_epochs);
    
    for k = 1:n_epochs
        X_real = EEG.data(:,:,k)'; 
        X_fft = fft(X_real, n_points, 1);
        X_analytic = ifft(X_fft .* h, [], 1);
        phase_data(:,:,k) = angle(X_analytic)';
    end
    
    % --- Step B: Calculate Phase Locking Value (PLV) ---
    fprintf('   -> Computing 30x30 PLV connectivity matrix...\n');
    PLV_matrix = zeros(NUM_CHANNELS, NUM_CHANNELS, n_points);
    
    for t = 1:n_points
        % Extract phase across all epochs for the current time point t
        phase_at_t = squeeze(phase_data(:, t, :)); % Size: [30 x n_epochs]
        
        for i = 1:NUM_CHANNELS
            for j = 1:NUM_CHANNELS
                phase_diff = phase_at_t(i, :) - phase_at_t(j, :);
                PLV_matrix(i, j, t) = abs(mean(exp(1i * phase_diff)));
            end
        end
    end
    
    % --- Step C: Append to Grand Tensor ---
    % Ensure time dimension matches (in case of slight data variations)
    t_limit = min(NUM_TIMEPOINTS, n_points);
    GROUP_TENSOR(:,:,s,1:t_limit) = PLV_matrix(:,:,1:t_limit);
    
    fprintf('   -> [%s] Successfully added to Group Tensor!\n', subj_id);
end

%% --- 4. SAVE FINAL TENSOR ---
fprintf('\n======================================================\n');
fprintf('   ALL SUBJECTS PROCESSED. SAVING DYNAMIC TENSOR...\n');
fprintf('======================================================\n');

final_save_path = fullfile(OUTPUT_DIR, 'Group_Dynamic_Tensor.mat');
save(final_save_path, 'GROUP_TENSOR', '-v7.3');

fprintf('[SUCCESS] Master Tensor saved to: %s\n', final_save_path);
