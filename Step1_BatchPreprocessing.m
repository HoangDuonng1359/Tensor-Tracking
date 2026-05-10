%% =========================================================================
% EEG PREPROCESSING PIPELINE - BATCH PROCESSING FOR 40 SUBJECTS
% Description: Automates Epoching, Filtering, and CSD for all subjects.
% =========================================================================
clear; clc; close all;

% --- 1. CONFIGURATION ---
DATA_PATH = 'D:\Data\';
SAVE_PATH = fullfile(DATA_PATH, 'preprocessed_data');

if ~exist(SAVE_PATH, 'dir')
    mkdir(SAVE_PATH);
    fprintf('[INFO] Created new directory: %s\n', SAVE_PATH);
end

NUM_SUBJECTS = 40;
RESPONSE_MARKERS = [111, 112, 121, 122, 211, 212, 221, 222];

fprintf('======================================================\n');
fprintf('   STARTING BATCH PREPROCESSING FOR %d SUBJECTS\n', NUM_SUBJECTS);
fprintf('======================================================\n\n');

%% --- 2. GLOBAL SPATIAL SETUP (Calculated only once) ---
fprintf('[GLOBAL SETUP] Initializing G and H spatial matrices...\n');

% Use sub-001 as the template to extract 10-05 coordinate system
template_file = fullfile(DATA_PATH, 'sub-001', 'eeg', 'sub-001_task-ERN_eeg.set');
if ~exist(template_file, 'file')
    error('Template file not found: %s. Please check your DATA_PATH.', template_file);
end

EEG_temp = pop_loadset('filename', 'sub-001_task-ERN_eeg.set', 'filepath', fullfile(DATA_PATH, 'sub-001', 'eeg'));

% Remove eye channels to match the 30-channel brain network
EEG_temp = pop_select(EEG_temp, 'nochannel', {'HEOG_left', 'HEOG_right', 'VEOG_lower'});
EEG_temp = pop_chanedit(EEG_temp, 'lookup', 'standard_1005.elc');

% Extract spherical coordinates and generate G, H matrices
X = [EEG_temp.chanlocs.X]; Y = [EEG_temp.chanlocs.Y]; Z = [EEG_temp.chanlocs.Z];
[theta, phi] = cart2sph(X, Y, Z);
M = struct('theta', rad2deg(theta(:)), 'phi', rad2deg(phi(:)), 'lab', {{EEG_temp.chanlocs.labels}'}, 'n', length(X));

[G, H] = GetGH(M);
fprintf('-> Successfully generated G and H matrices for %d channels.\n\n', M.n);

%% --- 3. SUBJECT LOOP PROCESSING ---
for s = 1:NUM_SUBJECTS
    % Format subject ID (e.g., sub-001, sub-012)
    subj_id = sprintf('sub-%03d', s);
    subj_dir = fullfile(DATA_PATH, subj_id, 'eeg');
    file_name = sprintf('%s_task-ERN_eeg.set', subj_id);
    file_path = fullfile(subj_dir, file_name);
    
    fprintf('------------------------------------------------------\n');
    fprintf('[%s] Processing started...\n', subj_id);
    
    % Safely check if subject data exists to prevent loop crashes
    if ~exist(file_path, 'file')
        fprintf('   [WARNING] File not found for %s. Skipping to next subject.\n', subj_id);
        continue; 
    end
    
    % --- Step A: Load, Epoch, and Filter ---
    % 1. Load dataset
    EEG = pop_loadset('filename', file_name, 'filepath', subj_dir);
    
    % 2. Extract 2-second epochs [-1 1] around response markers
    EEG = pop_epoch(EEG, num2cell(RESPONSE_MARKERS), [-1 1]);
    
    % 3. Remove baseline using the [-1000 -500] ms window
    EEG = pop_rmbase(EEG, [-1000 -500]);
    
    % 4. Bandpass filter for Theta frequency (4-8 Hz)
    EEG = pop_eegfiltnew(EEG, 4, 8);
    
    % --- Step B: Spatial Filtering (CSD) ---
    % 1. Ensure eye channels are removed before applying CSD
    if size(EEG.data, 1) == 33
        EEG = pop_select(EEG, 'nochannel', {'HEOG_left', 'HEOG_right', 'VEOG_lower'});
        EEG = pop_chanedit(EEG, 'lookup', 'standard_1005.elc');
    end
    
    [n_chan, n_points, n_epochs] = size(EEG.data);
    fprintf('   -> Extracted %d epochs (%d channels, %d timepoints).\n', n_epochs, n_chan, n_points);
    fprintf('   -> Applying Spherical Spline CSD...\n');
    
    % 2. Apply CSD epoch by epoch
    csd_data = zeros(n_chan, n_points, n_epochs);
    for k = 1:n_epochs
        csd_data(:,:,k) = CSD(EEG.data(:,:,k), G, H);
    end
    EEG.data = csd_data; % Update EEG structure with sharpened data
    
    % --- Step C: Save Intermediate Data ---
    % Saving individual .mat files is crucial to prevent RAM overflow
    save_name = sprintf('%s_Preprocessed.mat', subj_id);
    save_path = fullfile(SAVE_PATH, sprintf('%s_Preprocessed.mat', subj_id));
    
    fprintf('   -> Saving preprocessed data to disk...\n');
    save(save_path, 'EEG', '-v7.3');
    fprintf('   -> [DONE] Saved as %s\n', save_name);
end

fprintf('\n======================================================\n');
fprintf('   BATCH PREPROCESSING COMPLETED SUCCESSFULLY!\n');
fprintf('======================================================\n');