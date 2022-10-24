function y = sthresh(x,lambda)
%STHRESH Perform soft thresholding. 
%   Y = STHRESH(X,LAMBDA) returns a soft thresholded version of the input X
%   where the returned value is either the original value scaled toward 
%   zero (if above threhsold) or zero (if below threhold). LAMBDA>0 is the
%   threshold value.  Soft thresholding is also called 'shrinkage'.
%
%   If X is a vector or matrix, HTHRESH acts on each component separately.
%   A scalar LAMBDA is used for each entry and a vector (or matrix) LAMBDA
%   of the same size as X applies a a different threshold value to each
%   entry.

%   Author: Christopher J. Rozell (crozell@ece.gatech.edu) 
%   Creation date: 11/8/2007
%   Modification date: 11/9/2007

if (length(lambda) ~= 1)
   if ndims(x) == ndims(lambda)
       if min(size(x) == size(lambda))==0
           error('LAMBDA must be either a scalar or the same size as X.')
       end
   else
           error('LAMBDA must be either a scalar or the same size as X.')
   end
end

lambda = abs(lambda);
y   = (x - sign(x).*lambda).*(abs(x) > lambda);
