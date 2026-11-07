
function [allPSD, psdInfo] = CHECK_JOINT_PSD(K_F, K_FY, K_T, tol)
% CHECK_JOINT_PSD
% Independently check whether all joint feature-label
% kernel matrices are positive semi-definite (PSD).
%
% Inputs:
%   K_F  : 1xV cell, feature kernel matrices (d_v x d_v)
%   K_FY : 1xV cell, cross-space kernel matrices (d_v x c)
%   K_T  : c x c, label kernel matrix
%   tol  : numerical tolerance (default: 1e-10)

    if nargin < 4 || isempty(tol)
        tol = 1e-10;
    end

    %% 1. Convert single-view inputs to cell arrays

    if ~iscell(K_F)
        K_F = {K_F};
    end

    if ~iscell(K_FY)
        K_FY = {K_FY};
    end

    V = numel(K_F);

    if numel(K_FY) ~= V
        error('The numbers of K_F and K_FY views do not match.');
    end

    if size(K_T,1) ~= size(K_T,2)
        error('K_T must be a square matrix.');
    end

    %% 2. Initialize results

    psdInfo = repmat(struct( ...
        'minEigenvalue', NaN, ...
        'symmetryError', NaN, ...
        'threshold', NaN, ...
        'isPSD', false), V, 1);

    %% 3. Check each joint kernel matrix

    for v = 1:V

        KX  = K_F{v};
        KXY = K_FY{v};

        % Dimension checking
        if size(KX,1) ~= size(KX,2)
            error('K_F{%d} must be square.', v);
        end

        if size(KXY,1) ~= size(KX,1) || ...
           size(KXY,2) ~= size(K_T,1)
            error('Dimension mismatch in view %d.', v);
        end

        % Check finite values
        if any(~isfinite(KX(:))) || ...
           any(~isfinite(KXY(:))) || ...
           any(~isfinite(K_T(:)))
            error('NaN or Inf detected in view %d.', v);
        end

        % Construct joint kernel matrix
        M = [KX, KXY;
             KXY', K_T];

        % Check symmetry
        symmetryError = norm(M-M','fro') / ...
                        max(1,norm(M,'fro'));

        % Symmetric part for numerical eigenvalue calculation
        M_sym = (M+M')/2;

        % Eigenvalue decomposition
        eigenvalues = eig(M_sym);
        eigenvalues = real(eigenvalues);

        minEig = min(eigenvalues);

        % Relative numerical tolerance
        threshold = tol * max(1,max(abs(eigenvalues)));

        % PSD judgment
        isPSD = (symmetryError <= tol) && ...
                (minEig >= -threshold);

        % Save results
        psdInfo(v).minEigenvalue = minEig;
        psdInfo(v).symmetryError = symmetryError;
        psdInfo(v).threshold = threshold;
        psdInfo(v).isPSD = isPSD;

    end

    %% 4. Overall judgment

    allPSD = all([psdInfo.isPSD]);

end
