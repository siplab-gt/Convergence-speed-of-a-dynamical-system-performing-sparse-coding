% This code generates the figures in the paper "Convergence Speed of a
% Dynamical System for Sparse Recovery" by Aurèle Balavoine, Christopher J.
% Rozell and Justin K. Romberg

set(0,'DefaultLineLineWidth',2); 
set(0,'DefaultLineMarkersize',10); 
set(0,'DefaultTextFontsize',16); 
set(0,'DefaultAxesFontsize',16); 
set(0,'defaultaxeslinewidth',1);

%% Figure 2 and 3: Vary the signal sparsity S and threshold \lambda

clear all

% To generate the same figures as the paper:
rand('seed',8148)
randn('seed',9058);

% Number of values of S and lambda
k= 99;
% sparsity of the input
ss=floor(linspace(2,100,k));
% dimension of the input
N=400;
% number of measurements
M=200;
% Threshold
lambdaa=linspace(0.01,0.2,k);
%number of trials per value of S
nb = 100;
% cell of results for fig 2
u = zeros(k,k);
% cell of results for fig 3
q = zeros(k,k);
% Initial state for the LCA
init=zeros(N,1);

w = waitbar(0,'simulations running');
maxit = k*k*nb;
currit = 0;
tic
for i=1:k
    s=ss(i);  
    for j=1:k
        lambda = lambdaa(j);
        for l=1:nb
            % create sparse vector
            a0=zeros(N,1);
            ind=randperm(N);
            ind=ind(1:s);
            val=sign(randn(1,s)); % assign random signs
            a0(ind)=val/(norm(val));

            % Create random matrix of measurement
            Phi=rand(M,N);
            Phi = orth(Phi')';
            for jj=1:N
                Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
            end

            % Create measurement vector
            y0=Phi*a0;
            y0=y0(:);
            sigma=0;
            y=y0+sigma*randn(M,1);

            % solve L1 with LCA
            [a1, u1, reconMSE, coefCost, numCoef, time, trueRelErr, error,...
            maxSupp, ut, opt] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
            init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

            u(j,i) = u(j,i) + opt;
            q(j,i) = q(j,i) + max(numCoef);
            
            currit = currit+1;
            esttime =  round((maxit - currit)*(toc/currit)/60);
            waitbar(currit/maxit,w,['Estimated time remaining is ', num2str(esttime), ' minute(s)']);
        end
    end   
end
close(w)
u = u/nb;
q = q/nb;

%% Figure 2:
% Percentage of the trials where no more than the S nodes from the
% optimal support \Gamma_{\dagger} become active during convergence. The value 1 means
% that 100% of the trials satisfied this condition.

figure('units','pixels','Position',[50 50 600 500]);
imagesc(ss,lambdaa,u)
set(gca, 'YDir', 'normal')
axis square;
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 30 480 480]);
xlabel('sparsity S')
ylabel('threshold \lambda')
colormap(hot)
colorbar

%% Figure 3:
% Ratio of the maximum number of active element q during convergence
% over the sparsity level S. For instance, a value of 10 in the color bar means
% that the biggest active set during convergence contains 10S active elements.

qratio = q;
for i=1:k
    qratio(:,i) = qratio(:,i)/ss(i);
end
figure('units','pixels','Position',[50 50 600 500]);
imagesc(ss,lambdaa,log10(qratio))
set(gca, 'YDir', 'normal')
axis square;
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 30 480 480]);
xlabel('sparsity S')
ylabel('threshold \lambda')
%title('Ratio of maximum support size over sparsity level: q/S')
colormap(flipud(hot))
h = colorbar;
minval = min(get(h,'Ytick'));
tvals = [0.01 0.1 1 10 100];
colorbar('Ytick', [minval log10(tvals)], 'YTickLabel', ['0 |',sprintf('%g |',[tvals])])

%% Figure (deleted from the paper):
% Evolution of the nodes with time. Nodes in the optimal support are plotted
% in plain, and nodes in its complement are dashed. For a large enough
% value of the threshold, only nodes in the optimal support become active.
% As the threshold is decreased, more nodes enter the active set.

clear all

% To generate the same figures as the paper:
rand('seed',1863)
randn('seed',8626);

% sparsity of the input
s=5;
% dimension of the input
N=400;
% number of measurements
M=200;
% Vector with the values of the threshold to test
k = 3;
lambdaa = [0.05, 0.1, 0.2];

% Generates the s-sparse original signal a0 with s entries with value +/- 1
% selected at random:
a0=zeros(N,1);
ind=randperm(N);
ind=ind(1:s);
val=1;
% val = randi(10,1,s); % <- can put randi(K,1,s) here to create s random values
                  % between 1 and K instead of having amplitude 1.
val=val.*sign(randn(1,s)); % assign random signs
a0(ind)=val/(norm(val)); % normalize the original vector to have unit norm.

figure
stem(ind,a0(ind),'xr')
xlim([1 N])
xlabel('node number')
ylabel('amplitude')
title('Original vector a_0')

% Generates the Gaussian random matrix for measurement:
Phi=1/M*rand(M,N);
Phi = orth(Phi')';
for i=1:N
    Phi(:,i)=Phi(:,i)./norm(Phi(:,i)); % normalize the columns to 1
end

% Noiseless vector of measurements:
y0=Phi*a0;
% Noise variance:
sigma = 0.025;
% Noisy vector of measurements:
y=y0+sigma*randn(M,1);

figure
plot(y0,'r')
hold on
plot(y,'--')
xlim([1 M])
xlabel('sample')
ylabel('input y')
legend('signal','signal+noise','location','best')

% Set the initial state for the LCA algorithm
init=zeros(N,1);

% Indices of the nodes in the optimal support
act=ind(:);
% Indices of the nodes not in the optimal support
inact=(1:N)'; inact(ind)=[];

% Make the plots for the different values of lambda
%  Set the color table
colors = fireprint(N+50);

for i=1:k
    lambda = lambdaa(i);
    
    % Solve the LCA with the current threshold
    [aLCA, uLCA, reconMSE, coefCost, numCoef, time, trueRelErr, error,...
        MaxSupp, ut] = ...
        SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], init, 500, ...
        0.001, 0.01, [1e-4, .1], a0, @L1Cost, []);
    
    % Number of iterations
    NumIt=size(ut,1);
    
    figure('units','pixels','Position',[50 50 500 500])
    hold on
    for j=1:s
        plot(time,ut(:,act(j)),'Color',colors(j,:),'Linewidth',1);
    end
    for j=1:N-s
        plot(time,ut(:,inact(j)),'--','Color',colors(j+s,:),'Linewidth',1);
    end
    plot(time,lambda*ones(NumIt,1),'k','Linewidth',1)
    plot(time,-lambda*ones(NumIt,1),'k','Linewidth',1)
    xlim([0 time(end)])
    axis square
    axis tight
    v=axis;
    axis([v(1) v(2) v(3)-0.01 v(4)+0.01])
    xlabel('number of time constants \tau')
    ylabel('u(t)')
    %txt = annotation('textbox',[0.5,0.75,0.23,0.08],'String',['\lambda= ' num2str(lambda)]);
    %box on
    title(['Thereshold \lambda = ', num2str(lambda)])
end

%% Figure (deleted from the paper):
% Plot in loglog scale of the maximum number of active elements
% q as a function of the threshold lambda, averaged over 100 trials.

clear all

% To generate the same figures as the paper:
rand('seed',7870)
randn('seed',6018);

% sparsity of the input
s=5;
% dimension of the input
N=400;
% number of measurements
M=200;
% Number of data points
k=100;
lambdaa=linspace(0.01,0.2,k);
%number of trials per point
nb = 100;
% cell of results containing the average maximum size of the active set
u = zeros(1,k);
% Initial state for the LCA
init=zeros(N,1);

w = waitbar(0,'simulations running');

for i=1:k
    lambda=lambdaa(i);  
    for j=1:nb
        % create sparse vector
        a0=zeros(N,1);
        ind=randperm(N);
        ind=ind(1:s);
        val=sign(randn(1,s)); % assign random signs
        a0(ind)=val/(norm(val));
        
        % Create random matrix of measurement
        Phi=rand(M,N);
        Phi = orth(Phi')';
        for jj=1:N
            Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
        end
        
        % Create measurement vector
        y0=Phi*a0;
        y0=y0(:);
        sigma=0.025;
        y=y0+sigma*randn(M,1);

        % solve L1 with LCA
        [a1, u1, MSE, coefCost, numCoef] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
        init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);
        
        u(i) = u(i)+ max(numCoef);
        
        waitbar((nb*(i-1)+j)/(k*nb),w);
    end
end
close(w)
u = u/nb;

figure('units','pixels','Position',[50 50 600 500]);
loglog(lambdaa,u,'k')
xlabel('threshold \lambda')
ylabel('size q of the active set')
ylim([min(u)-1 max(u)+10])
xlim([lambdaa(1),0.2])

%% Figure 4: Exponentially decreasing the threshold
% This figure shows the number of active nodes (left column) and the
% fixed point a_* reached by the LCA (right column), for different choices of
% the threshold. The red crosses represent the original signal a^{\dagger} and the blue
% rounds are the solutions a_*. A fixed threshold  \lambda = 0.3 was used in the first
% row, \lambda = 0.08 in the second row, and the threshold was decreased from 0.3
% to 0.08 according to an exponential decay in the third row.

clear all

% To generate the same figures as the paper:
rand('seed',4539)
randn('seed',1438); %1438 4458

% sparsity of the input
s=5;
% dimension of the input
N=400;
% number of measurements
M=200;

% create sparse vector
a0=zeros(N,1);
ind=randperm(N);
ind=ind(1:s);
val=randn(1,s);
a0(ind)=val/(norm(val));

figure
stem(ind,a0(ind),'xr')
xlim([1 N])
xlabel('node number')
ylabel('amplitude')
title('Original vector a_0')

% Create random matrix of measurement
Phi=rand(M,N);
Phi = orth(Phi')';
for jj=1:N
    Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
end

% Create measurement vector
y0=Phi*a0;
y0=y0(:);
sigma=0.025;
y=y0+sigma*randn(M,1);

% ----- Initial state for the LCA
init=zeros(N,1);

% solve L1-min with LCA for fixed threshold
lambda1 = 0.3;

[a1, u1, MSE, coefCost, numCoef1, time1] = SolveLCA_CS(y, Phi,[], @sthresh, lambda1, [], ...
init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

% solve L1-min with LCA for fixed threshold
lambda2 = 0.08;

[a2, u2, MSE, coefCost, numCoef2, time2] = SolveLCA_CS(y, Phi,[], @sthresh, lambda2, [], ...
init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

% solve L1-min with LCA for decaying threshold
L = length(numCoef2);
lambdav = calcGradThresh(lambda1,lambda2,0.001,0.01,L);
lambdav=[lambdav,lambda2*ones(1,500-L)];

[av, uv, MSE, coefCost, numCoefv, timev] = SolveLCA_CS(y, Phi,[], @sthresh, lambdav, [], ...
init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

% ----- Plot
xmax = max(max(time1(end),time2(end)),timev(end));
ymax = max([numCoef1,numCoef2,numCoefv]);
ylima0 = [min(a0(ind))-0.1 max(a0(ind))+0.1];

% figure('units','normalized','Position',[.1 .1 .7 .6])
figure('units','pixels','Position',[20 80 900 600])

splot = subplot(3,2,1);
plot(time1,[0,numCoef1])
xlim([0 xmax])
ylim([0 ymax+1])
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])

splot = subplot(3,2,2);
inds=find(a1);
stem(inds,a1(inds))
hold on
stem(ind,a0(ind),'-xr')
ylim(ylima0)
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])
set(splot,'YTick',[-1:0.5:1])

splot = subplot(3,2,3);
plot(time2,[0,numCoef2])
xlim([0 xmax])
ylim([0 ymax+1])
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])

splot = subplot(3,2,4);
inds=find(a2);
stem(inds,a2(inds))
hold on
stem(ind,a0(ind),'-xr')
ylim(ylima0)
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])
set(splot,'YTick',[-1:0.5:1])

splot = subplot(3,2,5);
plot(timev,[0,numCoefv])
xlim([0 xmax])
ylim([0 ymax+1])
xlabel('number of time constants \tau')
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])

splot = subplot(3,2,6);
inds=find(av);
stem(inds,av(inds))
hold on
stem(ind,a0(ind),'-xr')
ylim(ylima0)
xlabel('node number')
set(splot,'Position',get(splot,'Position') + [-0.04 0 0.04 0.02])
set(splot,'YTick',[-1:0.5:1])

ylab = suplabel('size q of the active set','y');
xlab = suplabel('amplitudes of a_0 and a_*','yy');

set(ylab,'Position',get(ylab,'Position') + [.04 0 0 0]);
set(xlab,'Position',get(xlab,'Position') - [.05 0 0 0]);

%% Figure 5-(a): Vary the signal length N

clear all

% To generate the same figures as the paper:
rand('seed',4438)
randn('seed',7254);

% Number of values of N
k = 5;
% sparsity of the input
s=5;
% dimension of the input
NN=floor(linspace(200,1000,k));
% number of measurements
M=200;
% Thereshold value
lambda=0.1;
%number of trials per value of N
nb = 100;
% cell of results
u = cell(k,1);
% Theoretical speed
deltaExp = sqrt(s*log(NN/s)/M); %sqrt(s*log(NN)/M);

% Set plot parameters
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
Name=cell(k);
for i=1:k
    Name{i}=['N= ' num2str(NN(i))];
end
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
w = waitbar(0,'simulations running');
maxtime=0;

for i=1:k
    N=NN(i);  
    init=zeros(N,1);
    for j=1:nb
        % create sparse vector
        a0=zeros(N,1);
        ind=randperm(N);
        ind=ind(1:s);
        val=sign(randn(1,s)); % assign random signs
        a0(ind)=val/(norm(val));

        % Create random matrix of measurement
        Phi=rand(M,N);
        Phi = orth(Phi')';
        for jj=1:N
            Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
        end

        % Create measurement vector
        y0=Phi*a0;
        y0=y0(:);
        sigma=0.025;
        y=y0+sigma*randn(M,1);

        % solve L1-min with LCA
        [a1, u1] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
        init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

        [a1, u1, n1, n2, n3, time1, n4, u1Norm, maxSupp1] = SolveLCA_CS(y, ...
        Phi,[], @sthresh, lambda, [], init, 500, 0.001, 0.01, ...
        [1e-4, .1], a1, @L1Cost, [], 0, u1);
    
        l1 = length(u{i});
        l2 = length(u1Norm);
        if isempty(u{i})
            u{i} = u1Norm/u1Norm(1)/nb;
        else
            u{i} = [u{i},(u{i}(end))*ones(1,l2-l1)];
            u1Norm = [u1Norm,u1Norm(end)*ones(1,l1-l2)];
            u{i} = u{i} + u1Norm/u1Norm(1)/nb;
        end
        waitbar((nb*(i-1)+j)/(k*nb),w);
    end
    
    % plot results
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
    
    if time1(end)>maxtime
        maxtime=time1(end);
    end
end
close(w);

% Plot
time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),'--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off'); 
end
xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% replot
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),...
        '--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off'); % Exclude line from legend
    hold on
end
for i=1:k
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
end

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% Figure 5-(b): Vary the signal sparsity S

clear all

% To generate the same figures as the paper:
rand('seed',1179)
randn('seed',1745);

% Number of values of S
k=5;
% sparsity of the input
ss=floor(linspace(2,20,5));
% dimension of the input
N=400;
% number of measurements
M=200;
% Threshold
lambda=0.1;
%number of trials per value of S
nb = 100;
% cell of results
u = cell(k,1);
% Theoretical speed
deltaExp=sqrt(ss.*log(N./ss)/M); %deltaExp=sqrt(ss.*log(N)/M);
% Initial state for the LCA
init=zeros(N,1);

% Set plot parameters
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
 Name={};
 for i=1:k
   Name{i}=['s= ' num2str(ss(i))];
 end
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
w = waitbar(0,'simulations running');
maxtime=0;

for i=1:k
    s=ss(i);  
    for j=1:nb
        % create sparse vector
        a0=zeros(N,1);
        ind=randperm(N);
        ind=ind(1:s);
        val=sign(randn(1,s)); % assign random signs
        a0(ind)=val/(norm(val));

        % Create random matrix of measurement
        Phi=rand(M,N);
        Phi = orth(Phi')';
        for jj=1:N
            Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
        end

        % Create measurement vector
        y0=Phi*a0;
        y0=y0(:);
        sigma=0.025;
        y=y0+sigma*randn(M,1);

        % solve L1 with LCA
        [a1, u1] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
        init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

        [a1, u1, n1, n2, n3, time1, n4, u1Norm, maxSupp1] = SolveLCA_CS(y, ...
        Phi,[], @sthresh, lambda, [], init, 500, 0.001, 0.01, ...
        [1e-4, .1], a1, @L1Cost, [], 0, u1);
        l1 = length(u{i});
        l2 = length(u1Norm);
        if isempty(u{i})
            u{i} = u1Norm/u1Norm(1)/nb;
        else
            u{i} = [u{i},(u{i}(end))*ones(1,l2-l1)];
            u1Norm = [u1Norm,u1Norm(end)*ones(1,l1-l2)];
            u{i} = u{i} + u1Norm/u1Norm(1)/nb;
        end
        waitbar((nb*(i-1)+j)/(k*nb),w);
    end
    
    % plot results
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
    if time1(end)>maxtime
        maxtime=time1(end);
    end
end
close(w)

time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),...
      '--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
     'IconDisplayStyle','off'); % Exclude line from legend
end

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% replot
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),...
        '--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off'); % Exclude line from legend
    hold on
end
for i=1:k
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
end

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% Figure 5-(c): Vary the number of measurements M

clear all

% To generate the same figures as the paper:
rand('seed',4373)
randn('seed',5244);

% Number of values of M
k=5;
% sparsity of the input
s=5;
% dimension of the input
N=400;
% number of measurements
MM=floor(linspace(100,400,k));
% Threshold
lambda=0.1;
%number of trials per value of M
nb = 100;
% cell of results
u = cell(k,1);
% Theoretical speed
deltaExp=sqrt(s*log(N/s)./MM); % deltaExp=sqrt(s*log(N)./MM);
% Initial state of the LCA
init=zeros(N,1);

% Set plot parameters
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
Name={};
 for i=1:k
   Name{i}=['M= ' num2str(MM(i))];
 end
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
w = waitbar(0,'simulations running');
maxtime=0;

for i=1:k
    M=MM(i);  
    for j=1:nb
        % create sparse vector
        a0=zeros(N,1);
        ind=randperm(N);
        ind=ind(1:s);
        val=sign(randn(1,s)); % assign random signs
        a0(ind)=val/(norm(val));

        % Create random matrix of measurement
        Phi=rand(M,N);
        Phi = orth(Phi')';
        for jj=1:N
            Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
        end

        % Create measurement vector
        y0=Phi*a0;
        y0=y0(:);
        sigma=0.025;
        y=y0+sigma*randn(M,1);

        % solve L1 with LCA
        [a1, u1] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
        init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

        [a1, u1, n1, n2, n3, time1, n4, u1Norm, maxSupp1] = SolveLCA_CS(y, ...
        Phi,[], @sthresh, lambda, [], init, 500, 0.001, 0.01, ...
        [1e-4, .1], a1, @L1Cost, [], 0, u1);
        l1 = length(u{i});
        l2 = length(u1Norm);
        if isempty(u{i})
            u{i} = u1Norm/u1Norm(1)/nb;
        else
            u{i} = [u{i},(u{i}(end))*ones(1,l2-l1)];
            u1Norm = [u1Norm,u1Norm(end)*ones(1,l1-l2)];
            u{i} = u{i} + u1Norm/u1Norm(1)/nb;
        end
        waitbar((nb*(i-1)+j)/(k*nb),w);
    end
    
    % plot results
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
    if time1(end)>maxtime
        maxtime = time1(end);
    end
end
close(w)
time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),...
        '--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off'); % Exclude line from legend
end
xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% replot
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
time1=linspace(0,maxtime,5);
for i=1:k
    p3=semilogy(time1,exp(-(1-deltaExp(i))*time1),...
        '--','Color',colors(cval(i),:),'Markersize',8);
    set(get(get(p3,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off'); % Exclude line from legend
    hold on
end
for i=1:k
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
end

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% Figure 5-(d): Vary the threshold lambda

clear all

% To generate the same figures as the paper:
rand('seed',2022)
randn('seed',6512);

% Number of values of lambda
k=5;
% sparsity of the input
s=5;
% dimension of the input
N=400;
% number of measurements
M=200;
% lambdaa=linspace(0.025,0.1,5);
lambdaa=[0.02,0.025,0.03,0.06,0.1];
%number of trials per value of lambda
nb = 100;
% cell of results
u = cell(k,1);
% average maximum size of active set
MaxNact = zeros(k,1);
% Theoretical speed
deltaExp=sqrt(s*log(N/s)/M); %sqrt(s*log(N)/M);
% Initial state for the LCA
init=zeros(N,1);

% Set plot parameters
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
 Name={};
 for i=1:k
   Name{i}=['\lambda= ' num2str(lambdaa(i))];
 end
% figure('units','normalized','Position',[.1 .1 .6 .6]);
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
w = waitbar(0,'simulations running');
maxtime=0;

for i=1:k
    lambda=lambdaa(i);  
    for j=1:nb
        % create sparse vector
        a0=zeros(N,1);
        ind=randperm(N);
        ind=ind(1:s);
        val=sign(randn(1,s)); % assign random signs
        a0(ind)=val/(norm(val));

        % Create random matrix of measurement
        Phi=rand(M,N);
        Phi = orth(Phi')';
        for jj=1:N
            Phi(:,jj)=Phi(:,jj)./norm(Phi(:,jj)); % normalize the columns to 1
        end

        % Create measurement vector
        y0=Phi*a0;
        sigma=0.025;
        y=y0+sigma*randn(M,1);

        % solve L1 with LCA
        [a1, u1] = SolveLCA_CS(y, Phi,[], @sthresh, lambda, [], ...
        init, 500, 0.001, 0.01, [1e-4, .1], a0, @L1Cost);

        [a1, u1, n1, n2, numCoef, time1, n4, u1Norm, maxSupp1] = SolveLCA_CS(y, ...
        Phi,[], @sthresh, lambda, [], init, 500, 0.001, 0.01, ...
        [1e-4, .1], a1, @L1Cost, [], 0, u1);
        l1 = length(u{i});
        l2 = length(u1Norm);
        if isempty(u{i})
            u{i} = u1Norm/u1Norm(1)/nb;
        else
            u{i} = [u{i},(u{i}(end))*ones(1,l2-l1)];
            u1Norm = [u1Norm,u1Norm(end)*ones(1,l1-l2)];
            u{i} = u{i} + u1Norm/u1Norm(1)/nb;
        end
        MaxNact(i) = MaxNact(i) + max(numCoef);
        waitbar((nb*(i-1)+j)/(k*nb),w);
    end
    % plot results
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
    
    if time1(end)>maxtime
        maxtime=time1(end);
    end
end
close(w)
MaxNact = MaxNact/nb;

time1=linspace(0,maxtime,5);
p3=semilogy(time1,exp(-(1-deltaExp)*time1),...
    '--','Color',colors(3*k,:),'Markersize',8);
set(get(get(p3,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off'); % Exclude line from legend
p3=semilogy(time1,exp(-(1-sqrt(5)*deltaExp)*time1),...
    '--','Color',colors(5,:),'Markersize',8);
set(get(get(p3,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off'); % Exclude line from legend

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

%% replot
colors=fireprint(3*k,'invert',1);
cval = floor(exp(log(4)+(1:k)*(log(3*k)-log(4))/k));
figure('units','pixels','Position',[50 50 600 580]);
figaxes = gca();
set(figaxes,'units','pixels','Position',[90 70 480 480]);
for i=1:k
    time1=(0:length(u{i})-1)*0.001/0.01;
    p1=semilogy(time1,u{i},'Color',colors(cval(i),:),...
        'DisplayName',Name{i});
    hold on;
end
time1=linspace(0,maxtime,5);
p3=semilogy(time1,exp(-(1-deltaExp)*time1),'--', ...
    'Color',colors(3*k,:),'Markersize',8);
set(get(get(p3,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off'); 
p3=semilogy(time1,exp(-(1-sqrt(5)*deltaExp)*time1),...
'--','Color',colors(5,:),'Markersize',8);
set(get(get(p3,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off'); % Exclude line from legend

xlabel('number of time constants \tau');
hYlab = ylabel('$||u-u^*||_2$','Fontsize',18);
set(hYlab, 'Interpreter', 'latex');
xlim([0,maxtime]);
ylim([u1Norm(end)/u1Norm(1) 1])
axis square
leg=legend('show', 'location','southwest');

