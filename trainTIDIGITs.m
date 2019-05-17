clc;clear;close;
dbstop if error;
 
%% Import Speech Dataset
load('TIDIGIT_train.mat','train_samples','train_labels'); % 2464
load('TIDIGIT_test.mat','test_samples','test_labels');% 2486

samples = [train_samples;test_samples];
labels = [train_labels;test_labels];
RandomSqc = randperm(numel(samples));
train_samples = samples(RandomSqc(1:end-1000));
train_labels = labels(RandomSqc(1:end-1000));
test_samples = samples(RandomSqc(end-1000:end));
test_labels = labels(RandomSqc(end-1000:end));

%% Spectrogram Analysis to generate Filter Banks and MFCC vectors
cd mfcc; % switch to mfcc directory
[FBETrainList,MFCCTrainList] = MFCCDigit(train_samples);
[FBETestList,MFCCTestList] = MFCCDigit(test_samples);

channelMax = max(cell2mat(FBETrainList'),[],2); % max element along each row
channelMin = min(cell2mat(FBETrainList'),[],2); % min element along each row
%% Latency Coding %%
cd ..;  %switch to home directory

FBESpikeTrain = latencyCode(FBETrainList,channelMax,channelMin);
FBESpikeTest = latencyCode(FBETestList,channelMax,channelMin);

% FBESpikeTrain = FBETrainList;
% FBESpikeTest = FBETestList;
%% Self-Organizing Map(SOM) for feature extraction
dim1 = 24;
dim2 = 24;
if exist('SOMNet.mat','file')
    load('SOMNet.mat');
else
    net = SOM([FBESpikeTrain;FBESpikeTest],dim1,dim2);
    save('SOMNet','net'); 
end

% Prepare SOM Outputs for training set by passing the train samples through 
% trained SOM to generate outputs(Best Matching Unit)
nTrain = length(FBESpikeTrain);% number of training samples
DigitSOMTrainOutput = cell(nTrain,1);
K = 1; % number of firing neuron per frame

for idx = 1:nTrain
    input = FBESpikeTrain{idx};
    label = somTest(input,net,K);
    DigitSOMTrainOutput{idx} = label;
end

% Prepare SOM Outputs for testing set by passing the train samples through 
% trained SOM to generate outputs(Best Matching Unit)
nTest =  length(FBESpikeTest); % number of testing samples
DigitSOMTestOutput = cell(nTest,1);

for idx = 1:nTest
    input = FBESpikeTest{idx};
    label = somTest(input,net,K);
    DigitSOMTestOutput{idx} = label;
end

%% Spike Train Generation
dt = 1e-3;
nAfferents = dim1*dim2; % number of output neurons in SOM
[ptnTrain,TmaxTrain] = SpikeGenerator(DigitSOMTrainOutput,dt,nAfferents);
[ptnTest,TmaxTest] = SpikeGenerator(DigitSOMTestOutput,dt,nAfferents);

%% Supervised Tempotron Learning
nCls = 11; % number of output classes
nRun = 1;
AllWeights = zeros(nAfferents,nCls,1);

% Train SNN with Tempotron learning rule
for iRun = 1:nRun 
    weight = GnrlTmptrClsTr_L2(nAfferents, nTrain, nCls, TmaxTrain, ptnTrain,cell2mat(train_labels)');
    AllWeights = AllWeights + weight;
end

AllWeights = AllWeights./nRun;

save('weightTempotron','AllWeights');
save('testSet','ptnTest','TmaxTest','nTest','test_labels');
save('trainSet','ptnTrain','TmaxTrain','nTrain','train_labels');
