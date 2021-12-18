function [A, B] = covMatern5ard(logtheta, x, z)

% Matern covariance function with nu = d/2, d = 5 and with Automatic Relevance
% Determination (ARD) distance measure.
%
% Modified from covMatern5iso.m of GPML toolbox version 1.3 (copyright (c) 
% 2006 by Carl Edward Rasmussen, 2006-03-20)
% 
% The covariance function is:
%
%   k(x,z) = f( sqrt(d)*r ) * exp(-sqrt(d)*r)
%
% with f(t) = 1+t+t^2/3 since we are in the case of d=5.
%
% Here r is the distance sqrt((x-z)'*inv(P)*(x-z)), where the P matrix
% is diagonal with ARD parameters ell_1^2,...,ell_D^2, where D is the dimension
% of the input space and sf2 is the signal variance. The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          ..
%         log(ell_D)
%         log(sf) ]

d = 5;

if nargin == 0, A = '(D+1)'; return; end          % report number of parameters

[n, D] = size(x);

ell = exp(logtheta(1:D));                         % characteristic length scale
sf2 = exp(2*logtheta(D+1));                                   % signal variance

f = @(t) 1 + t.*(1+t/3); df = @(t) (1+t)/3;  % functions for d = 5
m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);

if nargin == 2                                      % compute covariance matrix
    K = sq_dist(diag(sqrt(d)./ell)*x');
    A = sf2*m(sqrt(K),f);
elseif nargout == 2                              % compute test set covariances
    A = sf2;
    K = sq_dist(diag(sqrt(d)./ell)*x',diag(sqrt(d)./ell)*z');
    B = sf2*m(sqrt(K),f);
else                                            % compute derivative matrices
    K = sq_dist(diag(sqrt(d)./ell)*x');
    if z <= D
        Ki = sq_dist(sqrt(d)/ell(z)*x(:,z)');
        A = sf2*dm(sqrt(K),f).*Ki;
        A(Ki<1e-12) = 0;
    else                                            % magnitude parameters
        A = 2*sf2*m(sqrt(K),f);
    end
end
