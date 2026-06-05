marketSPY = hist_stock_data_period('01071996','01012013','1wk','history','SPY');
% Why not use nasdaq composite ^IXIC as market indicator, it has 3300
% stocks (because you cannot invest in it?.)

% Dow jones (^DJI / DIA) has 30 stocks

% ABALX has 1700 stocks  

% nasdaq composite 100 (QQQ) 

% if we stay in USD area, we can use 
% S&P is split into industries by different funds
% then we can see how industry performs in comparison to the market.

% split into high-tech companies or consumer companies. 

% it might make sense to look at the individual chosen etfs, and see
% if some are biased in some industries.
% maybe QQQ performs best, and we should add some information about
% why it performs the best. 

% Do they have smaller stocks, larger stocks etc. 
% choose 7 that we are comfortable with, and we know what they are based
% off. We should consider which one would perform better than the other.

% QQQ is not the most diversified ETF given that it owns only non-financial companies and is heavily weighted to just a handful. 
funds = {'QQQ', 'DIA', 'ABALX', 'FOCPX', 'FSCHX'};
if ~exist('stocks', 'var')
    stocks=hist_stock_data('01012002','01012022', funds, 'frequency','mo');
end
N = length(stocks);
W = 10; % number of windows - 1
[~,yr_market] = computeReturns(marketSPY);

market_return = mean(yr_market);

[year_returns,expected_returns] = listReturns(stocks);

figure
hold on
for i=0:W
    returns = year_returns(:,(1+i):(10+i));
    exp_returns = mean(returns);
    [exp_return_p,variance_p] = efficientFrontier(10,cov(returns),0.05,0.02,exp_returns);
    plot(sqrt(variance_p),exp_return_p)
    [minstd,I] = min(sqrt(variance_p));
    text(minstd, exp_return_p(I),""+(i+1))
    disp(i+1)
    disp("covariance")
    disp(funds)
    disp(cov(returns))
    disp("correlation")
    disp(funds)
    disp(corr(returns))
    disp("mean return")
    disp(funds)
    disp(exp_returns)
end
legend('1','2','3','4','5','6','7','8','9','10','11')
hold off

figure
hold on
for i=0:W
    returns = year_returns(:,(1+i):(10+i))';
    exp_returns = mean(returns);
    [slope,x,y,exp_return_p,variance_p,X] = efficientFrontierSBL(cov(returns),0.01,exp_returns');
    plot(x,y)
    scatter(sqrt(variance_p),exp_return_p,'r')
    disp(i+1)
    disp('Holdings')
    disp(funds)
    disp(X')
    text(0.5+i/100,(0.5+i/100)*slope+0.01,""+(i+1))
end
legend('1','','2','','3','','4','','5','','6','','7','','8','','9','','10','','11','')
hold off


% FOR E)
% fix stakeholder, no matter the risk, i want 10 or 15 percent return
% each year.
% you have to deviate from efficient frontier each year, because
% the efficient portfolio can deviate on each year.
% we have to force portfolio for our goal each year.
% choose each year the efficient portfolio, is the basic strategy

% but it deviates our goal, so we might have to reduce or increase
% gain. 

% assume that you can borrow and lend. Tobin seperation means borrowing
% and lending is allowed

% download conversion rate last 20 years.
% every single month you convert currency.
% average return for each month

% X [1,N] with N fractions of x stock
% B [1,N] betas of portfolio
% A [1,N] alphas of portfolio
% sigma2_e [N,1] with N sigmas
% market [N,1] yearly returns.
function [expectation,variance] = SingleIndexModelStats(X,B,A,sigma2_e,M)
    market_exp = mean(M);
    market_var = var(M);
    beta_portfolio = X*B';
    variance = (beta_portfolio^2)*market_var+(X.^2)*sigma2_e';
    expectation = X*A' + beta_portfolio*market_exp;
end

% S [1,N] stock returns
% M [1,N] market returns
function [beta, alpha, sigma2_e] = SingleIndexModel(S,M)
    market_return = mean(M);
    stock_return = mean(S);
    diffM = M - market_return;
    diffS = S - stock_return;
    sigmaSM = diffS*diffM';
    sigmaM = diffM*diffM';
    beta = sigmaSM/sigmaM;
    alpha = stock_return - beta*market_return;
    diffPrediction = S - (alpha+beta*M);
    sigma2_e = (1/length(S))*(diffPrediction*diffPrediction');
end

% Ss [N,M] N different stocks, with M returns.
% M [1,N] market returns
function [betas, alphas, sigmas2_e] = SingleIndexModelGroup(Ss,M)
    [N, ~] = size(Ss);
    betas = zeros(1,N);
    alphas = zeros(1,N);
    sigmas2_e = zeros(1,N);
    for i=1:N
        [beta, alpha, sigma2_e] = SingleIndexModel(Ss(i,:),M);
        betas(i) = beta;
        alphas(i) = alpha;
        sigmas2_e(i) = sigma2_e;
    end
end

% Treynor & Mauzy regression
% rate is the risk free rate
% stock returns is [1,T] stock returns for T periods
% market returns is [1,T] stock returns for T periods
function coefficients = TMReg(rate,stock_returns,market_returns)
    excess_stock_returns = stock_returns - rate;
    excess_market_returns = market_returns - rate;
    T = length(stock_returns);
    X = ones(T,3);
    X(:,2) = excess_market_returns';
    X(:,3) = (excess_market_returns.^2)';
    coefficients = (X'*X)\(X'*excess_stock_returns');
end

% Note this works on returns not excess returns.
% betas [1,N] N different betas
% variance of residual risk [1,N] N different variances
% expected returns [1,N] market returns
% yearly returns [N,T], with N stocks and T time periods
function gammas = CSReg(betas,sigmas2_e,expected_returns)
    N = length(betas);
    X = ones(N,3);
    X(:,2) = betas';
    X(:,3) = sigmas2_e';
    gammas = (X'*X)\(X'*expected_returns');
end

function [exp_return_p,variance_p] = efficientFrontier(erate,covariance,r1,r2,R)

    excess_return1 = R - r1;
    excess_return2 = R - r2;
    
    Z1 = linsolve(covariance,[excess_return1,excess_return2]);

    A = [1 r1;1 r2];

    C1 = linsolve(A,Z1');
    
    pins = 10000;
    rates = linspace(-20,erate,pins);
    newA = ones(pins,2);
    newA(:,2) = rates';

    Z = newA*C1;
    X = Z./sum(Z,2);
    exp_return_p = X*R;
    variance_p = zeros(pins,1);
    for i=1:pins
        Xm = X(i,:);
        variance_p(i) = Xm*covariance*Xm';
    end
    

end

% Covariance (N,N) covariances
% risk free interest rate
% expected returm [N,1]
function [slope,x,y,exp_return_p,variance_p,X] = efficientFrontierSBL(covariance,r,R)

    excess_return = R - r;
    
    Z = linsolve(covariance,excess_return);
    
    X = Z/sum(Z); % maybe not use abs value

    exp_return_p = X'*R;
    variance_p = X'*covariance*X;
    
    slope = (exp_return_p - r)/sqrt(variance_p);
    
    x = linspace(0,0.9);
    y = slope*x + r;
end

function [year_returns,expected_returns] = listReturns(stocks)
    [~,fst_return] = computeReturns(stocks(1));
    D = length(fst_return);
    N = length(stocks);
    year_returns = zeros(N,D);
    expected_returns = repelem(0.0,N);
    for i=1:N
        [~,yr_return] = computeReturns(stocks(i));
        year_returns(i,:) = yr_return;
        expected_returns(i) = mean(yr_return);
    end
end

function r = logReturn(X,Y)
    r = log(X./Y);
end

function yreturn = yearlyReturns(X)
    yreturn = 1:length(X);    
    for k=1:length(X)
        curyear=X{k};
        yreturn(k) = sum(curyear);
    end
end

function [r,yearly_r] = computeReturns(X)
    X_year = arrayfun(@year,X.Date);
    gc = groupcounts(X_year);
    gc(1) = gc(1) - 1; %remove for other method
    X_X = X.AdjClose(2:end); %X.AdjClose;
    X_Y = X.AdjClose(1:(end-1)); %circshift(X.AdjClose,1);
    %X_Y(1) = X.Open(1)*X.AdjClose(1)/X.Close(1);
    r = logReturn(X_X,X_Y);
    split = mat2cell(r, gc); 
    yearly_r = yearlyReturns(split);
end