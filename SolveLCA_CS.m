
function [x, u, reconMSE, coefCost, numCoef, time, trueRelErr, error,...
    maxSupp, ut, opt] = SolveLCA_CS(y, A, AFuncArgs, actFunc, lambda, ...
    actFuncArgs, initState, maxIter, stepSize, tau, tol, xTrue, ...
    costFunc, costFuncArgs, plotStep, uTrue)

% SolveLCA_CT: Approximate solution to the continuous-time LCA algorithm.
%
% Usage
%   [x, u, reconMSE, coefCost, numCoef, trueRelErr] = SolveLCA_CT(y, A, AFuncArgs, actFunc, ...
%       actFuncArgs, initState, maxIter, stepSize, tau, tol, xTrue, costFunc, costFuncArgs,plotStep)
%
% Input
%       y               signal vector of length n
%       A               implicit (function handle) or explicit 
%                           (n by m matrix) dictionary
%       AFuncArgs       function arguments for implicit A function (ignored
%                           for explicit A functions)
%       actFunc         activation function
%       lambda          tradeoff parameter (threshold value)
%       actFuncArgs     activation function arguments
%       initState       initial length-n state (u) vector of the LCA
%                           (default is length-n zero vector)
%       maxIter         maximum number of iterations to perform (default=300) 
%       stepSize        step size for numerical approximation to LCA
%                           dynamical system (default=1e-3)
%       tau             time constant of dynamic system (default=1e-2)
%       tol             tolerance on the state (u) vector and the reconstruction 
%                           error(stopping criteria) (defult=[1e-4 0.2])
%       xTrue           true value of x to compare to LCA estimate (optional, default=[])
%       costFunc        coefficient cost function (optional, default=[])
%       costFuncArgs    coefficient cost function arguments (optional, default=[])
%       plotStep        Frequency of plotting the LCA status (default=0)
%
% Outputs
%       x               length m vector solution of the LCA
%       u               length m vector state of the LCA (x=actFunc(u))
%       mse             mean squared error as a function of the iteration number
%       coefCost        coefficient cost as a function of the iteration number
%       numCoef         number of active coefficients as a function of the
%                       iteration number 
%       trueRelErr      if xTrue is specified, this is the relative error
%                       as a function of the iteration number
%
% Description
%   SolveLCA_CT is an approximate implementation of the continuous time
%   LCA system 
%
%   du/dt = 1/tau (A'*y - u - (A'*A-I)*x) with x = actFunc(u, actFuncArgs)
%
%   to estimate the solution of  min ||A*x-y||^2 + lambda*sum(C(x_i)).
%
%   The cost function C depends on the form of actFunc.  Two specific cases
%   of interest are: actFunc is a hard threhsold function making C equal to
%   the L0 norm, and actFunc is a soft threhsold function making C equal to
%   the L1 norm.  The latter case (C=L1 norm) is a convex program, meaning
%   that x is a globally optimal solution. This is not true in the former
%   case.
%
%   The matrix A can be either an explicit (n by m) matrix containing
%   dictionary elements on the rows, or an implicit operator implemented as
%   a function. If using the implicit form, the user should provide an
%   implicit function of the following format:
%       Aout = A(mode, Ain, AFuncArgs)
%   This function gets as input a vector Ain and returns the equivalent of
%   Aout = A*Ain if mode = 1, or Aout = A'*Ain if mode = 2.  Ain is a
%   vector of length m when mode = 1, or a vector of length n when mode =
%   2.  The arguments AFuncArgs are passed directly to the function.
%
%   lambda is a tradeoff parameter between the reconstruction MSE and the
%   sparsity of the solution.  It often represents the threshold value for
%   the activation function, and will be passed to actFunc (see below).  To
%   facilitate schemes that use a different threshold for each node or
%   graduate the threshold over each iteration, lambda can take many forms.
%   If lambda is a vector, the kth iteration will use curLambda=lambda(k).  
%   If lambda is an array,  the kth iteration will use the vector 
%   curLambda=lambda(:,k) to impose a different threshold on each node.  
%   When lambda is non-scalar, the length of the argument serves as the 
%   maximum number of iterations (overriding maxIter if specified).

%   actFunc is a function handle to the activation function for the LCA.  
%   Special cases of particular interest are the hard or the soft
%   thresholding functions (see hthresh.m and sthresh.m).  When actFuncArgs 
%   exist (i.e., not NULL), they are passed to actFunc, and can take several 
%   forms.  If actFunc is a cell array, the kth iteration will use 
%   curActFuncArgs=actFuncArgs{k}.  If actFunc is a vector, the kth iteration 
%   will use curActFuncArgs=actFuncArgs(k).  If actFunc is an array,  the kth
%   iteration will use curActFuncArgs=actFuncArgs(:,k).  It is called as: x
%   = actFunc(u, curLambda, curActFuncArgs).  When actFuncArgs is non-scalar, 
%   the length of the argument serves as the maximum number of iterations 
%   (overriding maxIter if specified).
%
%   initState is the length m vector denoting the starting state (u) of the
%   LCA dynamical system.  With time-varying signals, it may be
%   desireable to have each time-step start with an initial state that is
%   the finishing state from the previous time step.  If nothing is
%   specified for initState, the default is a length-m zero vector.
%
%   maxIter is the maximum number of iterations for the LCA simulation.
%   If not specified, the default is maxIter=300.  If specified as an empty
%   element, maxIter=[], no iteration stopping criteria will be used.
%
%   stepSize denotes the temporal sampling rate of the discrete
%   approximation to the continuous time system.  If not specified, the
%   default is stepSize=1e-3.
%
%   tau is the time constant of the continuous time dyanamical system.  If
%   not specified, the default is tau=1e-2.
%
%   tol is a two dimensional vector specifying the tolerance on the state 
%   vector (u) and the gradient of the reconstruction error for the stopping 
%   condition.
%   If ((||u_{k-1} - u_{k}||_2)/(||u_{k-1}||_2) < tol(1)) and 
%   (||A'(y-Ax_{k})||_infinity)/min(lambda(:,end)))-1 < tol(2),
%   then the simulation stops.  min(lambda(:,end)) represents the minimum
%   value of the final threshold.  The first condition assures that the
%   system state is not changing much.  The second condition assures that
%   the gradient of the error term is small.  If only specified as a single 
%   scalar, the same value will be used for both conditions.  If not specified, 
%   the default is tol=[1e-4 0.2]. If specified as an empty element, tol=[], no 
%   tolerance stopping criteria will be used.
%
%   xTrue is the true value of x that generated the data y=xTrue.  When
%   this length-m vector is specified, the function returns a relative
%   error measure with each iteration.
%
%   costFunc is a function handle to the coefficient cost function for the LCA.  
%   It is called as: c = costFunc(x, costFuncArgs).  If this optional
%   argument is specified, the output coefCost will also be returned.
%
%   plotStep is the frequency of plots updating the LCA status (in number
%   of iteration steps between plots).  When plotStep=0, no plot updates
%   will be produced.  The default is plotStep=0.
%
%   x is a length-m vector containing the final output coefficients of the
%   LCA system.
%
%   u is a length-m vector containing the final state of the LCA system.
%
%   reconMSE is a vector containing the mean-squared reconstruction error, 
%   ||Ax-y||^2, of the LCA at each iteration.  The length of the vector
%   depends on how many iterations are performed until the stopping
%   criteria are met.
%
%   coefCost is a vector containing the coefficient cost, C(x), of the LCA
%   at each iteration.  The length of the vector depends on how many
%   iterations are performed until the stopping criteria are met.  This
%   output is only returned if the input argument costFunc is specified.
%
%   numCoef is a vector containing the number of active coefficients
%   used by the LCA at each iteration.  The length of the vector depends 
%   on how many iterations are performed until the stopping criteria are met.  
%
%   trueRelErr is a vector containing the relative error
%   norm(x-xTrue)/norm(xTrue) for each iteration. The length of the vector depends 
%   on how many iterations are performed until the stopping criteria are
%   met. 

%   Author: Christopher J. Rozell (crozell@ece.gatech.edu) 
%   Creation date: 11/13/2007
%   Modification date: 2/9/2008
%
%   Modified by: Aurèle Balavoine
%   Date: 11/15/2012


if (nargin<6)
    error('Must specify y, A, AFuncArgs, actFunc, lambda, and actFuncArgs.');
end

if (nargin<16)
    uTrue=[];
    error=[];
end
if (nargin<15)
    plotStep=0;
end
if (nargin<14)
    costFuncArgs=[];
end
if (nargin<13)
    costFunc=[];
end
if (nargin<12)
    xTrue=[];
    trueRelErr=[];
end
if (nargin<11)
    tol=[1e-4, .2];
end
if (nargin<10)
    tau=.01;
end
if (nargin<9)
    stepSize=.001;
end
if (nargin<8)
    maxIter=300;
end
if (nargin<7)
    initState = zeros(size(A,2));
end

%If only one tolerance number is specified, duplicate it for both entries
if ~isempty(tol)
    if(length(tol)==1)
        tol(2) = tol(1);
    end
end

%put the threshold in terms of function handle 'curLambda' so we don't have to
%care whether it was a scalar, vector or cell array
if  isscalar(lambda)
    curLambda = @(k) lambda;
elseif isvector(actFuncArgs)
    curLambda = @(k) lambda(k);
else
    curLambda = @(k) lambda(:,k);
end

%put the activation function arguments in terms of function handle
%'curActFunc' so we don't have to care whether it was a scalar, vector or cell
%array 
if iscell(actFuncArgs)
    curActFunc = @(k) actFuncArgs{k};
elseif isscalar(actFuncArgs)
    curActFunc = @(k) actFuncArgs;
elseif isvector(actFuncArgs)
    curActFunc = @(k) actFuncArgs(k);
else
    curActFunc = @(k) actFuncArgs(:,k);
end


%readjust maxIter if either lambda or actFuncArgs is spcified as a vector but
%it has less entries than the original maxIter
if (length(actFuncArgs) > 1)
    maxIter = min([maxIter length(actFuncArgs)]);
end

if (length(lambda) > 1)
    maxIter = min([maxIter length(lambda)]);
end

%put the activation function in terms of function handle 'actFunch' so we
%don't have to keep track of passing in the arguments and threshold
if (isempty(actFuncArgs))
    actFunch = @(u,k) actFunc(u, curLambda(k));
else
    actFunch = @(u,k) actFunc(u, curLambda(k), curActFunc(k));
end

%put everything in terms of function handle for the A 'matrix'.  Ah
%represents the synthesis transform (multiplication by A) and ATh 
%represents the analysis transform (multiplication by A')
if ~isa(A, 'function_handle')
   Ah = @(Ain, Args) A*Ain;
   ATh = @(Ain, Args) A'*Ain;
else
   Ah = @(Ain, Args) A(1,Ain,Args);
   ATh = @(Ain, Args) A(2,Ain,Args);
end

%set up first iteration
del = stepSize/tau;
u = initState;						%state variables
x = actFunch(u, 1);					%output (activation) variables
recon = Ah(x, AFuncArgs);			%current reconstruction
resid = y-recon;					%current residual
newproj = ATh(resid, AFuncArgs);	%projection of residual onto dictionary
gradf = -(newproj - (u - x));		%(warped) gradient of the objective function

contIter = 1;						%continue iterating?
countIter = 1;						%iteration counter

coefCost = [];
if ~isempty(uTrue)
    error(1) = norm(u-uTrue);
end
time(1) = 0;
maxSupp=(x~=0);
maxNum=sum(maxSupp);
ut(1,:)=u;
opt = 1;
if ~isempty(xTrue)
    ind = (xTrue ~=0);
    if (sum( (ind-maxSupp) ) < 0)
        opt = 0;
    end
end

while contIter
  
  %find new states/coefficients
  prevu = u;
  u = u - del*gradf;
  x = actFunch(u, countIter);
  recon = Ah(x, AFuncArgs);
  resid = y-recon;
  newproj = ATh(resid, AFuncArgs);
  gradf = -(newproj - (u - x));
  curSupp=(x~=0);
  
  %calculate the reconstruction error, cost function, and number of non-zeros
  reconMSE(countIter) = norm(resid)^2;
  if ~isempty(costFunc)
    coefCost(countIter) = costFunc(x, curLambda(countIter), costFuncArgs);
  end
  if ~isempty(xTrue)
    trueRelErr(countIter) = norm(x-xTrue)/norm(xTrue);
    if (sum( (ind-curSupp) ) < 0)
        opt = 0;
    end
  end
  if ~isempty(uTrue)
    error(countIter+1) = norm(u-uTrue);
  end
  time(countIter+1)=time(countIter)+del;
  numCoef(countIter) = sum(curSupp);
  if(numCoef(countIter)>maxNum)
      maxSupp=curSupp;
      maxNum=numCoef(countIter);
  end
  ut(countIter+1,:)=u;
  
  %calculate percentage change of the state variables and check convergence  
  perChange(countIter) = norm(prevu-u)/norm(prevu);
  if ~isempty(tol)
    if ((perChange(countIter)<tol(1)) && ( (max(abs(newproj))/curLambda(maxIter))-1 < tol(2) ))
        contIter=0;
    end
  end

  if ~isempty(maxIter)
    if (countIter >= maxIter)
        contIter=0;
    end
  end
  
  %plot results, if desired
  if (mod(countIter,plotStep)==0)
    plotLCAprogress(u, curLambda(countIter), reconMSE, coefCost,perChange,numCoef)
  end

  countIter = countIter+1;    

end
