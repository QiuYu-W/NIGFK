clear
clc
% 关闭所有警告信息
warning('off')
% 将当前目录下的所有子目录添加到搜索路径中
addpath(genpath(pwd));
tic;

params.lambda1  = 0.01; 
params.lambda2 = 1; 
params.beta   = 0.1; 
params.gamma  = 0.001; 

Train= load('yeast_2_view.mat');
[Anchors, ModelInfo] = Anchor_MVML(Train);

K_F = MI_FF(Train,Anchors);
K_FY = MI_FY(Train,Anchors);
K_T = MI_YY(Train,Anchors);

[allPSD, psdInfo] = CHECK_JOINT_PSD(K_F, K_FY, K_T);

if allPSD
    fprintf('All joint kernel matrices are PSD.\n');
    fprintf('Skip PSD.\n');
    K_X  = K_F;
    K_XY = K_FY;
    K_Y  = K_T;
else
    [K_X, K_XY, K_Y] = PSD_ALL(K_F, K_FY, K_T);
end

[feature_slc] = KMVML(K_X, K_XY, K_Y, params);