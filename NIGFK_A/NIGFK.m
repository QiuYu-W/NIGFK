function [feature_slc, running_time, savedState] = NIGFK(K_X, K_XY, K_Y, param)
% 目标函数:
% min_{W,H,A,A*} sum_v ( ||Phi(X^v)W^v - Phi(Y)A^v||_F^2 
%                      + lambda1 * ||Phi(X^v) - Phi(X^v)W^vH^v||_F^2
%                      + lambda2 * ||A^v - A^*||_F^2
%                      + gamma * ||W^v||_2,1 )
%                 + beta * ||Phi(Y)A^* - Phi(Y)||_F^2
%
% 输入:
%   K_X     : 1xV cell array, K_X{v} is d^v * d^v matrix (Feature Kernel)
%   K_XY    : 1xV cell array, K_XY{v} is d^v * c matrix (Cross Kernel)
%   K_Y     : c * c matrix (Label Kernel)
%   param   : 结构体, 包含以下字段:
%             param.lambda1 : 特征自重构系数
%             param.lambda2 : 共识约束系数
%             param.beta    : 标签对齐系数
%             param.gamma   : L2,1 稀疏系数
%             param.max_iter: 最大迭代次数
%             param.epsilon : 收敛阈值 
%
% 输出:
%   W           : 1xV cell array, 特征权重矩阵 (用于特征选择)
%   H           : 1xV cell array, 辅助重构矩阵
%   A_v         : 1xV cell array, 视图特异性标签嵌入
%   A_star      : 共识标签嵌入矩阵
%   obj_history : 目标函数值收敛曲线

    %% 1. 参数初始化与检查
    if nargin < 4
        error('必须提供 K_X, K_XY, K_Y 和 param');
    end
    
    lambda1 = param.lambda1;
    lambda2 = param.lambda2;
    beta    = param.beta;
    gamma   = param.gamma;
    max_iter = 300;
    if isfield(param, 'max_iter'), max_iter = param.max_iter; end
    epsilon = 1e-3;
    if isfield(param, 'epsilon'), epsilon = param.epsilon; end
    
    V = length(K_X);      % 视图数量
    c = size(K_Y, 1);     % 标签数量

    time_start = tic;
    
    % rng(8);                 % 固定随机种子保证可复现
    savedState = rng;
    
    % 初始化变量
    W = cell(1, V);
    H = cell(1, V);
    A_v = cell(1, V);
    
    % 随机初始化 W, H, A_v, A_star
    % 注意: W, H 需要非负，A 不需要
    for v = 1:V
        dv = size(K_X{v}, 1);
        W{v} = rand(dv, c);      
        H{v} = rand(c, dv);
        A_v{v} = rand(c, c); 
    end
    A_star = rand(c, c);
    
    obj_history = [];
    
    % A_star 的更新矩阵: (lambda2 * V * I + beta * K_Y)^-1
    Inv_Matrix_A_star = inv(lambda2 * V * eye(c) + beta * K_Y + 1e-6 * eye(c));
    
    % A_v 的更新矩阵: (K_Y + lambda2 * I)^-1
    Inv_Matrix_A_v = K_Y + lambda2 * eye(c) + 1e-6 * eye(c);
    

    %% 2. 迭代优化主循环
    for iter = 1:max_iter
        
        % --- 第一步: 更新 A^v (Closed-form) ---
        sum_Av = zeros(c, c);
        for v = 1:V
            Kxy = K_XY{v};
            % 公式: A^v = (Ky + lambda2*I)^-1 * (Kxy' * W + lambda2 * A*)
            A_v{v} = Inv_Matrix_A_v \ (Kxy' * W{v} + lambda2 * A_star);
            sum_Av = sum_Av + A_v{v};
        end

        % --- 第二步: 更新 A* (Closed-form) ---
        % 公式: A^* = (lambda2*V*I + beta*Ky)^-1 * (lambda2*sum(Av) + beta*Ky)
        % A_star = Inv_Matrix_A_star \ (lambda2 * sum_Av + beta * K_Y);
        A_star = Inv_Matrix_A_star * (lambda2 * sum_Av + beta * K_Y);
        
        % --- 视图并行更新 ---
        for v = 1:V
            % 获取当前视图数据
            Kx = K_X{v};
            Kxy = K_XY{v};
            dv = size(Kx, 1);
            
            % --- 第三步: 更新 H^v (Multiplicative Update) ---
            % 梯度正项(分母): W' * Kx * W * H
            % 梯度负项(分子): W' * Kx
            
            Term_H_Num = W{v}' * Kx; 
            Term_H_Den = W{v}' * Kx * W{v} * H{v};
            
            % 乘法更新 H (加eps防止除0)
            H{v} = H{v} .* (Term_H_Num ./ (Term_H_Den + 1e-9));
            
            % --- 第四步: 更新 W^v (Multiplicative Update with L2,1 norm) ---
            % 计算 L2,1 范数的导数矩阵 D (对角阵)
            % D_ii = 1 / (2 * ||w_i||_2)
            W_row_norms = sqrt(sum(W{v}.^2, 2));
            D_vec = 0.5 ./ (W_row_norms + 1e-9); % 避免除以0
            
            % 梯度正项(分母): Kx*W + lambda1*Kx*W*H*H' + gamma*D*W
            % 梯度负项(分子): Kxy*Av + lambda1*Kx*H'
            
            % 计算 D * W (利用广播机制加速，相当于对每行乘以 D_vec)
            DW = bsxfun(@times, D_vec, W{v});

            % 因为 A^v 存在负数，必须拆解以保证梯度正项和负项绝对非负
            Av_pos = max(A_v{v}, 0);  % (A^v)^+ 
            Av_neg = max(-A_v{v}, 0); % (A^v)^- 
            
            % 梯度负项(去分子): Kxy*(Av_pos) + lambda1*Kx*H'
            Term_W_Num = Kxy * Av_pos + lambda1 * Kx * H{v}';
            
            % 梯度正项(去分母): Kx*W + Kxy*(Av_neg) + lambda1*Kx*W*H*H' + gamma*D*W
            Term_W_Den = Kx * W{v} + Kxy * Av_neg + lambda1 * Kx * W{v} * H{v} * H{v}' + gamma * DW;
            
            % 乘法更新 W，严格保证非负性
            W{v} = W{v} .* (Term_W_Num ./ (Term_W_Den + 1e-9));
            
        end
        
        % --- 计算目标函数值 (用于检查收敛) ---
        current_obj = CalculateObjective(K_X, K_XY, K_Y, W, H, A_v, A_star, lambda1, lambda2, beta, gamma);
        obj_history = [obj_history; current_obj];
        
        % 收敛检查
        if iter > 1 && abs(obj_history(end-1) - current_obj) / abs(obj_history(end-1)) < epsilon
            % fprintf('算法在第 %d 次迭代收敛。\n', iter);
            break;
        end
    end

    running_time = toc(time_start);
    
    % 生成结果记录
    B = vertcat(W{:});
    % 根据行的L2范数进行特征排序
    [~, feature_slc] = sort(sqrt(sum(B.^2, 2)), 'descend');
end

function obj = CalculateObjective(K_X, K_XY, K_Y, W, H, A_v, A_star, l1, l2, beta, gamma)
    % 辅助函数：计算总目标函数值
    % 利用 trace 技巧计算核化后的 Frobenius 范数
    obj = 0;
    V = length(K_X);
    
    % 预计算与视图无关的项
    trKY = sum(K_Y(1:size(K_Y,1)+1:end));
    KY_Astar = K_Y * A_star;
    trAstar_KY_Astar = sum(sum(conj(A_star) .* KY_Astar));   % trace(A*'KY*A*)
    trAstar_KY = sum(sum(conj(A_star) .* K_Y));              % trace(A*'KY)
    
    for v = 1:V
        KX = K_X{v};
        KXY = K_XY{v};
        Wv = W{v};
        Hv = H{v};
        Av = A_v{v};
        
        % 复用中间结果，减少重复矩阵乘法
        KX_Wv = KX * Wv;
        KY_Av = K_Y * Av;
        KXY_Av = KXY * Av;
        
        % 1. ||Phi(X)W - Phi(Y)A||^2
        % = Tr(W'KX*W - 2*W'KXY*A + A'KY*A)
        term1 = sum(sum(conj(Wv) .* KX_Wv)) ...
              - 2 * sum(sum(conj(Wv) .* KXY_Av)) ...
              + sum(sum(conj(Av) .* KY_Av));
        
        % 2. ||Phi(X) - Phi(X)WH||^2
        % = Tr(KX - 2*KX*W*H + H'W'KX*W*H)

        trKX = sum(KX(1:size(KX,1)+1:end));
        term2_mid = sum(sum(KX_Wv .* (Hv.')));         % trace(KX*W*H)
        G = Wv' * KX_Wv;                               % W'KXW
        HHt = Hv * Hv';                               % H*H'
        term2 = trKX - 2 * term2_mid + sum(sum(G .* (HHt.')));  % trace(G*HHt)
        
        % 3. ||A - A*||^2 (这是普通矩阵，直接算)
        D = Av - A_star;
        term3 = sum(abs(D(:)).^2);
        
        % 4. L2,1 Norm
        term4 = sum(vecnorm(Wv, 2, 2));
        
        obj = obj + term1 + l1 * term2 + l2 * term3 + gamma * term4;
    end
    
    % 5. ||Phi(Y)A* - Phi(Y)||^2
    % = Tr(A*'KY*A* - 2*A*'KY + KY)
    term5 = trAstar_KY_Astar - 2 * trAstar_KY + trKY;
    
    obj = obj + beta * term5;
end
