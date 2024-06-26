% This is a sample demo code for Newton Method-based Subspace Support Vector Data Description
% Please contact fahad.sohrab@tuni.fi for any errors/bugs
clc
close all
clear

%% Possible inputs to nsvddtrain
% The first input argument is the Traindata (target training data)
% other inputs/options are
% params.C        :Value of hyperparameter C, Default=0.1.
% params.d        :Data in lower dimension, make sure that params.dim<D, Default=2.
% params.eta      :Used as step size for gradient, Default=0.01.
% params.npt      :Used for selecting non-linear data description. Possible options are 1 (for non-linear data description), default=1 (linear data description)
% params.s        :Hyperparameter for the kernel, used in non-linear data description. Default=10.
% params.minmax   :Possible options are 'max', 'min' ,Default='min'.
% params.maxIter  :Maximim iteraions of the algorithm. Default=10.
% params.consType :Regularizatioin term (0,1,2,3) Default=1,
% params.bta      :Controlling the importance of regularization term. Default=0.1.

%% Generate Random Data
noOfTrainData = 500; noOfTestData = 100;
D= 5; %D=Original dimensionality of data/features
Traindata = rand(D,noOfTrainData); %Training data/features
%Training labels (all +1s) are not needed.

testlabels = -ones(noOfTestData,1);
perm = randperm(noOfTestData);
positiveSamples = floor(noOfTestData/2);
testlabels(perm(1:positiveSamples))=1; % test labels, +1 for target, -1 for outliers
Testdata= rand(D,noOfTestData); %Testing data/features

%% Input parameters setting example
params.minmax = 'min';
params.maxIter = 5;
params.Cval=0.5;
params.d=2;
params.eta=0.2;
params.npt=1;
params.s=5;
params.maxIter = 10;
params.consType=3;
params.bta=0.1;

%% Training and Testing
nssvddmodel=nssvddtrain(Traindata,params);
[predicted_labels,eval]=nssvddtest(Testdata,testlabels,nssvddmodel);
