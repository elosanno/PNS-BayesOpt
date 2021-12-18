function [out1, out2] = igp(logtheta, covfunc, x, y, xstar)

% Training and prediction using multiple independent Gaussian processes with
% shared hyperparameters
% 
% Modified from gpr.m of GPML toolbox version 1.3 (copyright 2006 by Carl 
% Edward Rasmussen, 2006-03-20)
%
% Two modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = igp(logtheta, covfunc, x, y)
%    or: [Ypred Vpred]  = igp(logtheta, covfunc, x, y, xstar)
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   Ypred       is a (column) vector (of size nn) of prediced means
%   Vpred       is a (column) vector (of size nn) of predicted variances


if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed

[n, D] = size(x);
d = size(y,2); 

n_theta_x = eval(feval(covfunc{:}));
theta_x = logtheta(1:n_theta_x);
sigma2n = exp(2*logtheta(n_theta_x+1));  % Noise parameter
mu = logtheta(end);     % constant mean function 

% Deal with (independent) multivariate output by recursing
if size(y,2)>1
    % allocate
    if nargin==4, out1 = 0; out2 = zeros(length(logtheta),1); 
    else, out1 = zeros(size(xstar,1),d); out2 = zeros(size(xstar,1),d); end
    % GO
    for id = 1:d
        in = {logtheta, covfunc, x, y(:,id)};
        if nargin==5, in = {in{:}, xstar}; end 
        [out1_d,out2_d] = igp(in{:});   % perform inference for dimension number id of the output
        if nargin==4
            out1 = out1 + out1_d;   % sum nlZ
            if nargout==2                                                  
                out2 = out2 + out2_d;   % sum dnlZ
            end
        else    % concatenate mu and S2 for all the outputs 
            out1(:,id) = out1_d;
            out2(:,id) = out2_d;
        end
    end, return                                      % return to end the recursion
end


K = feval(covfunc{:}, theta_x, x);    % compute training set covariance matrix
K = K + sigma2n*eye(n); 

L = chol(K)';                        % cholesky factorization of the covariance

alpha = solve_chol(L',y-mu);

if nargin == 4 % if no test cases, compute the negative log marginal likelihood
    
    nl = 0.5*(y-mu)'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
    out1 = nl;
    
    if nargout == 2               % ... and if requested, its partial derivatives
        gradnl = zeros(size(logtheta));       % set the size of the derivative vector
        W = L'\(L\eye(n))-alpha*alpha';                % precompute for convenience
        for i = 1:n_theta_x
            gradnl(i) = sum(sum(W.*feval(covfunc{:}, theta_x, x, i)))/2;
        end
        gradnl(end-1) = sum(sum(W*2*sigma2n.*eye(n),2),1)/2;
        gradnl(end) = -sum(alpha);
        
        out2 = gradnl;
    end
     
else                    % ... otherwise compute (marginal) test predictions ...
    
    [Kss, Kstar] = feval(covfunc{:}, theta_x, x, xstar);     %  test covariances
    
    Ypred = mu + Kstar' * alpha;    % predicted means
    out1 = Ypred;
    
    if nargout == 2
        v = L\Kstar;
         Vpred = Kss - sum(v.*v,1)';    % predicted variances
         out2 = Vpred;
    end
    
end
