function [FBEList,MFCCList] = MFCCDigit(train_samples)
% Define variables
Tw = 50;                % analysis frame duration (ms)
Ts = 25;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 200;               % lower frequency limit (Hz)
HF = 5000;              % upper frequency limit (Hz)
fs = 20000;             % sampling rate (Hz)

% Train Set
num_of_sample = size(train_samples,1);
FBEList = cell(size(train_samples));
MFCCList = cell(size(train_samples));

for idx = 1:num_of_sample
	speech = train_samples{idx};

	% Feature extraction (feature vectors as columns)
	[ MFCCs, FBEs, ~ ] = ...
		mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
	
    %MFCCList{idx} =  MFCCs(2:end,:);
	MFCCList{idx} =  MFCCs;
    % FBEList{idx} = log(FBEs);
    FBEs  =  log(FBEs);
    FBEMean = mean(FBEs,2); % max element along each row
    FBEList{idx} =  bsxfun(@minus,FBEs,FBEMean);    
end
