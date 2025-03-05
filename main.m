clear;clc
close all

%% Add path
addpath('ClusteringMeasure')
addpath('funs')
addpath('data_M')
addpath('data_S')
addpath('data_L')

%% Parameters
load('MSRA25_uni');%MSRA25_uni,PalmData25_uni,isolet,usps,letter_uni
num_self = 20; %MSRA25_uni: 20; PalmData25_uni: 2; isolet: 64; usps: 256; letter_uni: 64;
numAnchor = 128; %MSRA25_uni: 128; PalmData25_uni: 512; isolet: 256; usps: 1024; letter_uni: 1024;
numNearestAnchor = 10; %MSRA25_uni: 10; PalmData25_uni: 90; isolet: 20; usps: 80; letter_uni: 50;
random_seed = 18; %MSRA25_uni: 18; PalmData25_uni: 15; isolet: 19; usps: 17; letter_uni: 8;

%% Init
X = full(X);
X = (double(X));
num = size(X,1);
X = X -repmat(mean(X),[num,1]);
if min(Y) == 0
  Y = Y + 1; 
end
c = length(unique(Y));

% 
[B,Anchor]=ULGEmzy(X,log2(numAnchor),numNearestAnchor,1); %1 BKHM,2 Kmeans++

%Randomly init F and G
[n,m] = size(B);
[~,F] = InitializeG_new(n,c,random_seed);
[~,G] = InitializeG_new(m,c,random_seed);

%% 
num_self = num_self*c;
[label_self,F_self,G_self,objs_self,iter_self,converge_self,t_self] = run_FSDGC(X,B,c,num_self,Y,F,G);
predictLabel = bestMap(Y,label_self);
result = ClusteringMeasure_new(Y, predictLabel);

fprintf('ACC：%.4f \n',result(1));
fprintf('NMI：%.4f \n',result(2));
fprintf('F-score：%.4f \n',result(4));
fprintf('ARI：%.4f \n',result(7));

     
    