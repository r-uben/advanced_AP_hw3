clc;
clear all;

%% INPUT: Calibration




%% OUTPUT: DYNARE ++

infl = inflation_prompt;

% Stochastic discount factor;
global delta gamma psi alpha theta;

delta = .999;
gamma = 10;
psi   = 1.5;
alpha = 1 - 1/psi;
theta = alpha / (1-gamma);

sdf_param_names     = ["delta" "gamma" "psi" "alpha" "theta"];
sdf_param_values    = [delta gamma psi alpha theta ];

% Cash flow;
global rho mu phi;
rho     = .98;
mu      = .0015; 
phi     = 2;

cash_flow_param_names   = ["rho" "mu" "phi"];
cash_flow_param_values  = [rho mu phi];

if infl

% Inflation
global sigma_q sigma_pi phi_q q_bar
    sigma_q = 0.35/100/sqrt(12);
    sigma_pi = 1.18/100/sqrt(12);
    phi_q = 0.78^(1/12);
    q_bar =  3.68*12/100;
    inflation_param_names = ["sigma_q" "sigma_pi" "phi_q" "q_bar"];
    inflation_param_values = [sigma_q sigma_pi phi_q q_bar];
end

% Shocks
global sigma_eps_c sigma_eps_x sigma_eps_d
sigma_eps_c = sqrt(3.6e-5);
sigma_eps_x = sqrt(9e-8);
sigma_eps_d = sqrt(0.001296);
if infl
    global sigma_eps_i;
    sigma_eps_i = 1;
end

% Steady states 
global mbar rfbar CEbar;
global pdbar pcbar ubar;
mbar = log(delta) - mu/psi;
pdbar = log(exp(mbar + mu)/(1 - exp(mbar + mu)));
pcbar = log(exp(mbar + mu)/(1 - exp(mbar + mu)));
ubar = (1/(1-1/psi))*log((1-delta)*(exp(pcbar)+1));
CEbar = exp((1 - gamma) * (ubar + mu));
rfbar = - mbar;

ss_param_names  = ["mbar" "pdbar" "pcbar" "ubar" "CEbar" "rfbar"];
ss_param_values = [mbar pdbar pcbar ubar CEbar rfbar] ;


% Write the .mod files to give them to dynare. In future versions, I think
% this should be included in an extra script/function to not make the code
% to big. Since we are asked to price bonds with maturities from 1 to 30,
% we should create 30 .mod files. However, we are going to do it first for
% maturities 1 and 2 and then generalise it once the code runs
% good.

prompt = "Maximum maturity: ";
Nmax = input(prompt); 
security = 'ZCE'; % First try ZCE;
mod = modFileGenerator;

% Model
equations = dictionary(string([]),string([]));
% Exogenous processes
equations("dc") = "dc = mu + x(-1) + eps_c;";
equations("x")  = "x = rho*x(-1) + eps_x;";
equations("dd") = "dd = mu + phi*x(-1) + eps_d;";
% Utility and SDF
equations("u") = "exp(alpha*u) = 1-delta+delta*exp(theta*ce);";
equations("m") = "exp(m) = delta*exp(-1/psi*dc)*" + ...
    "exp((-gamma+1/psi)*(u+dc))*exp((-1/psi+gamma)/(1-gamma)*ce(-1));";
equations("CE") = "CE = exp((1-gamma)*(u(+1)+dc(+1)));";
equations("ce") = "ce = log(CE);";
% Returns and ratios
equations("pc") = "exp(pc) = exp(m(+1)) * (1 + exp(pc(+1))) * exp(dc(+1));";
equations("pd") = "exp(pd) = exp(m(+1)) * (1 + exp(pd(+1))) * exp(dd(+1));";
equations("rf") = "1/exp(rf) = exp(m(+1));";
equations("rc") = "exp(rc) = (1+exp(pc))/exp(pc(-1))*exp(dc);";
equations("rm") = "exp(rm) = (1+exp(pd))/exp(pd(-1))*exp(dd);";
equations("exr") = "exr = rm-rf(-1);";
if infl
    equations("dpi") = "dpi = q(-1) + sigma_pi * eps_i;";
    equations("q") = "q = (1 - phi_q) * q_bar + phi_q * q(-1) + sigma_q * eps_i;";
end

% ZCE: General
pd_zce      = dictionary(string([]), string([]));
r_zce       = pd_zce;
exr_zce     = pd_zce;
% ZCE: Initial
pd_zce_0    = pd_zce;
r_zce_0     = pd_zce;
exr_zce_0   = pd_zce;
% B 
p_B         = dictionary(string([]), string([]));
r_B         = dictionary(string([]), string([]));
exr_B       = dictionary(string([]), string([])); 
% B: Initial
p_B_0     = pd_zce;
r_B_0     = pd_zce;
exr_B_0   = pd_zce;
if infl
    % B 
    p_NB         = dictionary(string([]), string([]));
    r_NB         = dictionary(string([]), string([]));
    exr_NB       = dictionary(string([]), string([])); 
    % B: Initial
    p_NB_0     = pd_zce;
    r_NB_0     = pd_zce;
    exr_NB_0   = pd_zce;
end

newline = ";\n";
% n=0

% First condition
pd_zce("pd_ZCE_0") = strjoin(["exp(pd_ZCE_0) = 1", newline]);
p_B("p_B_0") = strjoin(["exp(p_B_0) = 1", newline]);
if infl
    p_NB("p_NB_0") = strjoin(["exp(p_NB_0) = 1", newline]);
end
% Initial value
pd_zce_0("pd_ZCE_0_0") = strjoin(["pd_ZCE_0 = 0", newline]);
p_B_0("p_B_0_0") = strjoin(["p_B_0 = 0", newline]);
if infl
    p_NB_0("p_NB_0_0") = strjoin(["p_NB_0 = 0", newline]);
end

for n=1:Nmax
  % ZCE: Price-to-Dividend
  key = replace(strcat("pd_ZCE_", num2str(n)), ' ', '');
  pd_zce_LHS = strjoin(["exp(m(+1) + dd(+1) + pd_ZCE_", num2str(n-1), "(+1))"]);
  pd_zce_RHS = strjoin(["exp(pd_ZCE_", num2str(n), ")"]);
  pd_zce(key) = strrep(strjoin([pd_zce_LHS, "=", pd_zce_RHS, newline]), " ", "");
  pd_zce_0(init(key)) = strrep(strjoin([key, "=",num2str(n*(mu+rfbar)), newline]), " ", "");
  % ZCE: Returns
  key  = replace(strcat("r_ZCE_", num2str(n)), ' ', '');
  r_zce_LHS  = strjoin(["r_ZCE_", num2str(n)]);
  r_zce_RHS  = strjoin(["pd_ZCE_", num2str(n-1), " - pd_ZCE_", num2str(n), "(-1) +dd"]);
  r_zce(key)   = strrep(strjoin([r_zce_LHS, "=", r_zce_RHS, newline]), " ", "");
  r_zce_0(init(key)) = strrep(strjoin([key, "=", num2str(rfbar), newline]), " ", "");
  % ZCE: Excess Returns
  key = replace(strcat("exr_ZCE_", num2str(n)), ' ', '');
  exr_zce_LHS  = strjoin(["exr_ZCE_", num2str(n)]);
  exr_zce_RHS  = strjoin(["r_ZCE_", num2str(n), "-rf(-1)"]); 
  exr_zce(key) = strrep(strjoin([exr_zce_LHS, "=", exr_zce_RHS, newline]), " ", "");
  exr_zce_0(init(key)) = strrep(strjoin([key, "=0", newline]), " ", "");

  % B: Price
  key = replace(strcat("p_B_", num2str(n)), ' ', '');
  p_B_LHS   = strjoin(["exp(m(+1) + p_B_", num2str(n-1), "(+1))"]);
  p_B_RHS   = strjoin(["exp(p_B_", num2str(n), ")"]);
  p_B(key)  = strrep(strjoin([p_B_LHS, "=", p_B_RHS, newline]), " ", "");
  p_B_0(init(key)) = strrep(strjoin([key, "=", num2str(-n*rfbar), newline]), " ", "");
  % B: Returns
  key = replace(strcat("r_B_", num2str(n)), ' ', '');
  r_B_LHS    = strjoin(["r_B_", num2str(n)]);
  r_B_RHS    = strjoin(["p_B_", num2str(n-1), "- p_B_", num2str(n), "(-1)"]);
  r_B(key)   = strrep(strjoin([r_B_LHS, "=", r_B_RHS, newline]), " ", "");
  r_B_0(init(key)) = strrep(strjoin([key, "=",num2str(-rfbar),newline]), " ", "");
  % B: Excess Returns
  key = replace(strcat("exr_B_", num2str(n)), ' ', '');
  exr_B_LHS  = strjoin(["exr_B_", num2str(n)]);
  exr_B_RHS  = strjoin(["r_B_", num2str(n), "-rf(-1)"]);
  exr_B(key) = strrep(strjoin([exr_B_LHS, "=", exr_B_RHS, newline]), " ", "");
  exr_B_0(init(key)) = strrep(strjoin([key, "=0",newline]), " ", "");
  if infl
      % NB: Price
      key = replace(strcat("p_NB_", num2str(n)), ' ', '');
      p_NB_LHS   = strjoin(["exp(m(+1) + p_NB_", num2str(n-1), "(+1)-dpi(+1))"]);
      p_NB_RHS   = strjoin(["exp(p_NB_", num2str(n), ")"]);
      p_NB(key)  = strrep(strjoin([p_NB_LHS, "=", p_NB_RHS, newline]), " ", "");
      p_NB_0(init(key)) = strrep(strjoin([key, "= ", num2str(rfbar-q_bar),newline]), " ", "");
      % NB: Returns
      key = replace(strcat("r_NB_", num2str(n)), ' ', '');
      r_NB_LHS    = strjoin(["r_NB_", num2str(n)]);
      r_NB_RHS    = strjoin(["p_NB_", num2str(n-1), "- p_NB_", num2str(n), "(-1) -dpi"]);
      r_NB(key)   = strrep(strjoin([r_NB_LHS, "=", r_NB_RHS, newline]), " ", "");
      r_NB_0(init(key)) = strrep(strjoin([key, "=",num2str(-rfbar),newline]), " ", "");
      % BN: Excess Returns
      key = replace(strcat("exr_NB_", num2str(n)), ' ', '');
      exr_NB_LHS  = strjoin(["exr_NB_", num2str(n)]);
      exr_NB_RHS  = strjoin(["r_NB_", num2str(n), "-rf(-1)"]);
      exr_NB(key) = strrep(strjoin([exr_NB_LHS, "=", exr_NB_RHS, newline]), " ", "");
      exr_NB_0(init(key)) = strrep(strjoin([key, "=0",newline]), " ", "");
  end
end

% Variables' names
var_names = ["dc" "dd" "x" "pc" "pd" "u" "m" "CE" "ce" "rf" "rc" "rm" "exr"];
B_names = vertcat(p_B.keys, r_B.keys, exr_B.keys).';
zce_names =  vertcat(pd_zce.keys, r_zce.keys, exr_zce.keys).';

% Exogenous variables
exo_vars = ["eps_c" "eps_x" "eps_d"];


% Paramaters names and values in a dictionary
param_names = [sdf_param_names cash_flow_param_names];
param_values= [sdf_param_values cash_flow_param_values];
if infl
    param_names = [param_names inflation_param_names];
    param_values = [param_values inflation_param_values];
end
params      = dictionary(param_names, param_values);

% Initial values
var_bar_values = [mu mu 0 pcbar pdbar ubar mbar CEbar log(CEbar) rfbar rfbar rfbar 0];
initial_values = dictionary(var_names, var_bar_values);

if infl
    NB_names =  vertcat(p_NB.keys, r_NB.keys, exr_NB.keys).';
    var_names = [var_names ["q" "dpi"]];
    exo_vars  = [exo_vars "eps_i"];
    params("phi_q") = phi_q;
    params("sigma_q") = sigma_q;
    params("sigma_pi") = sigma_pi;
    initial_values("dpi") = q_bar;
    initial_values("q") = q_bar;
    sec_var_names =  cat(1, [B_names, zce_names, NB_names]);
else
    sec_var_names =  cat(1, [B_names, zce_names]);
end


% Name of the .mod file
filename = strcat('TS_BY04_RubenFernandez');
% Open the file to be writen:
fid = fopen(strcat(filename, '.mod'), 'w+');

mod.calibration(fid);

% Variables
fprintf(fid, '// Endogenous and exogenous variables \n');
mod.variables(fid, [var_names sec_var_names], exo_vars);

% Parameters
fprintf(fid,'// Parameters\n');
mod.parameters(fid, params);

% Model
fprintf(fid, "// Model\n");
fprintf(fid, "model; \n");
mod.model(fid, equations);
mod.security(fid, p_B, r_B, exr_B);
if infl
    mod.security(fid, p_NB, r_NB, exr_NB);
end
mod.security(fid, pd_zce, r_zce, exr_zce);
fprintf(fid, "end;\n\n");

% Initial Values
fprintf(fid, '// Initial Values\n');
fprintf(fid, "initval; \n");
mod.initial_values(fid, initial_values);
mod.sec_initial_values(fid, p_B_0, r_B_0, exr_B_0);
if infl
    mod.sec_initial_values(fid, p_NB_0, r_NB_0, exr_NB_0);
end
mod.sec_initial_values(fid, pd_zce_0, r_zce_0, exr_zce_0);
fprintf(fid, "end;\n\n");

% Variance-Covariance Matrix
if infl==0
fprintf(fid,strcat([...
'// Variance-Covariance Matrix\n',...
'//       SRS	LRS	   DivS  \n',...
'vcov =  [3.6e-5 0      0         \n',...
'         0      9e-8   0         \n',...
'         0      0      0.001296];\n\n',...
'// Approximation Order\n',...
'order = 2;']));
else
fprintf(fid,strcat([...
'// Variance-Covariance Matrix\n',...
'//       SRS	LRS	   DivS  Inflation\n',...
'vcov =  [3.6e-5 0      0         0\n',...
'         0      9e-8   0         0\n',...
'         0      0      0.001296  0\n',...
'         0      0      0         1];\n\n',...
'// Approximation Order\n',...
'order = 2;']));
end

%% OUTPUT: DYNARE ++
filename = 'TS_BY04_RubenFernandez';
eval(sprintf('!dpp/dynare++ --per 15 --sim 3 --ss-tol 1e-10 %s.mod',filename));
eval(sprintf('load %s.mat',filename));

results = table(dyn_vars, dyn_steady_states, dyn_ss);

%% SIMULATION

randn('state',2022);
N=Nmax;
prompt='Number of observations: ';
K=input(prompt); 
shocks = mvnrnd(zeros(1,length(exo_vars)), dyn_vcov_exo, K)'; % simulate shocks

sim = dynare_simul(filename, shocks, dyn_ss); 

dyn_i_p_B = [];
dyn_i_p_NB = [];
dyn_i_pd_ZCE = [];
for n=0:Nmax
    dyn_i_p_B_n = eval(['dyn_i_p_B_', num2str(n)]);
    dyn_i_pd_ZCE_n = eval(['dyn_i_pd_ZCE_', num2str(n)]);
    dyn_i_p_B    = [dyn_i_p_B dyn_i_p_B_n];
    dyn_i_pd_ZCE = [dyn_i_pd_ZCE dyn_i_pd_ZCE_n];
    if infl
        dyn_i_p_NB_n = eval(['dyn_i_p_NB_', num2str(n)]);
        dyn_i_p_NB    = [dyn_i_p_NB dyn_i_p_NB_n];
    end
end
% Simulations: Price of Bonds, Price Dividends of ZCE, Risk Free rate and
% Dividend Growth
dyn_i_p_B = sim(dyn_i_p_B, :);
dyn_i_pd_ZCE = sim(dyn_i_pd_ZCE, :);
if infl
        dyn_i_p_NB = sim(dyn_i_p_NB, :);
end
dyn_i_rf = sim(dyn_i_rf,:);
dyn_i_dd = sim(dyn_i_dd,:);
% Risk Premia for both bonds and ZCE:
rp_B = zeros(N, K);
rp_ZCE = zeros(N,K);
if infl
    rp_NB = zeros(N,K);
end
for t=2:K
    rf = dyn_i_rf(t-1);
    for n=1:N
        if n==1
            rp_B_n_t = -dyn_i_p_B(n, t-1)-rf;
            if infl
                rp_NB_n_t = -dyn_i_p_NB(n, t-1)-rf;
            end
            rp_ZCE_n_t = -dyn_i_pd_ZCE(1, t-1)+dyn_i_dd(t)-rf;
        else
            rp_B_n_t = dyn_i_p_B(n-1, t)-dyn_i_p_B(n, t-1)-rf;  
            rp_ZCE_n_t = dyn_i_pd_ZCE(n-1, t)-dyn_i_pd_ZCE(n, t-1)+dyn_i_dd(t)-rf; 
            if infl
                rp_NB_n_t = dyn_i_p_NB(n-1, t)-dyn_i_p_NB(n, t-1)-rf;  
            end
        end  
        rp_B(n,t) = rp_B_n_t;
        rp_ZCE(n,t) = rp_ZCE_n_t;
        if infl
            rp_NB(n,t) = rp_NB_n_t; 
        end
    end
end

rp_B_mean   = annualised_return(mean(rp_B,2)*100);
rp_B_vol    = annualised_std(std(rp_B,0,2)*100);
sp_B        = rp_B_mean ./ rp_B_vol;
rp_ZCE_mean = annualised_return(mean(rp_ZCE,2)*100);
rp_ZCE_vol  = annualised_std(std(rp_ZCE,0,2)*100);
sp_ZCE      = rp_ZCE_mean ./ rp_ZCE_vol;
if infl
rp_NB_mean   = annualised_return(mean(rp_NB,2)*100);
rp_NB_vol    = annualised_std(std(rp_NB,0,2)*100);
sp_NB        = rp_NB_mean ./ rp_NB_vol;
end

%% PLOTS

if infl
figure
subplot(3,1,1);
yyaxis left;
plot(1:N, rp_NB_mean, '--');
ylabel('Nominal Bond' , 'Interpreter', 'Latex');
yyaxis right;
plot(1:N, rp_ZCE_mean, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
title('Annualized Monthly Risk Premium: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');

subplot(3,1,2);
yyaxis left;
plot(1:N, rp_NB_vol, '--');
ylabel('Nominal Bond',  'Interpreter', 'Latex');
yyaxis right;
plot(1:N, rp_ZCE_vol, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
title('Annualized Monthly Volatility: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');

subplot(3,1, 3);
yyaxis left;
plot(1:N, sp_NB, '--');
ylabel('Nominal Bond', 'Interpreter', 'Latex');
yyaxis right;
plot(1:N, sp_ZCE, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
xlabel('Maturity (months)', 'Interpreter', 'Latex')
title('Annualized Monthly Sharpe Ratio: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');
 
exportgraphics(gcf,'secs/fig/maturities_with_infl.png','Resolution',300)
else
figure
subplot(3,1,1);
yyaxis left;
plot(1:N, rp_B_mean, '--');
ylabel('Real Bond' , 'Interpreter', 'Latex');
yyaxis right;
plot(1:N, rp_ZCE_mean, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
title('Annualized Monthly Risk Premium: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');

subplot(3,1,2);
yyaxis left;
plot(1:N, rp_B_vol, '--');
ylabel('Real Bond',  'Interpreter', 'Latex');
yyaxis right;
plot(1:N, rp_ZCE_vol, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
title('Annualized Monthly Volatility: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');

subplot(3,1, 3);
yyaxis left;
plot(1:N, sp_B, '--');
ylabel('Real Bond', 'Interpreter', 'Latex');
yyaxis right;
plot(1:N, sp_ZCE, '-');
ylabel('ZCE', 'Interpreter', 'Latex');
xlabel('Maturity (months)', 'Interpreter', 'Latex')
title('Annualized Monthly Sharpe Ratio: Bonds vs. ZCE (\%)', 'Interpreter', 'Latex');
 
exportgraphics(gcf,'secs/fig/maturities.png','Resolution',300)
end

%% TABLE
maturities=5:5:N;
M = length(maturities);
rp_B_mean_to_print = [];
rp_B_vol_to_print = [];
sp_B_to_print = [];
rp_NB_mean_to_print = [];
rp_NB_vol_to_print = [];
sp_NB_to_print = [];
rp_ZCE_mean_to_print = [];
rp_ZCE_vol_to_print = [];
sp_ZCE_to_print = [];
for i=1:M
    rp_B_mean_to_print = [rp_B_mean_to_print rp_B_mean(maturities(i))];
    rp_B_vol_to_print = [rp_B_vol_to_print rp_B_vol(maturities(i))];
    sp_B_to_print = [sp_B_to_print sp_B(maturities(i))];
    rp_NB_mean_to_print = [rp_NB_mean_to_print rp_NB_mean(maturities(i))];
    rp_NB_vol_to_print = [rp_NB_vol_to_print rp_NB_vol(maturities(i))];
    sp_NB_to_print = [sp_NB_to_print sp_NB(maturities(i))];
    rp_ZCE_mean_to_print = [rp_ZCE_mean_to_print rp_ZCE_mean(maturities(i))];
    rp_ZCE_vol_to_print = [rp_ZCE_vol_to_print rp_ZCE_vol(maturities(i))];
    sp_ZCE_to_print = [sp_ZCE_to_print sp_ZCE(maturities(i))];
end

table = tabConstructor();
caption = '';

% REAL BONDS
tab = table.open_it("results");
table.open_table_env(tab);
table.captionsetup(tab, 0.75);
table.captioning(tab, caption);
table.centering(tab);
table.row_sep(tab, 1.1);
table.open_tabular_env(tab, M);
table.panel(tab, "A", M);
table.write_row(tab, '$\E\left[r_{n,t}^{b,ex}\right]$', rp_B_mean_to_print);
table.write_row(tab, '$\sigma\hspace{-0.02cm}\left(r_{n,t}^{b,ex}\right)$', rp_B_vol_to_print);
table.write_row(tab, ['$\E\left[r_{n,t}^{b,ex}\right]/' ...
    '\sigma\hspace{-0.02cm}\left(r_{n,t}^{b,ex}\right)$'], sp_B_to_print);
table.panel(tab, "B", M);
table.write_row(tab, '$\E\left[r_{n,t}^{zce,ex}\right]$', rp_ZCE_mean_to_print);
table.write_row(tab, '$\sigma\hspace{-0.02cm}\left(r_{n,t}^{zce,ex}\right)$', rp_ZCE_vol_to_print);
table.write_row(tab, ['$\E\left[r_{n,t}^{zce,ex}\right]/' ...
    '\sigma\hspace{-0.02cm}\left(r_{n,t}^{zce,ex}\right)$'], sp_ZCE_to_print);
table.write_hline(tab);
table.write_hline(tab);
table.close_tabular_env(tab);
table.close_table_env(tab);

% NOMINAL BONDS
tab = table.open_it("results_with_infl");
table.open_table_env(tab);
table.captionsetup(tab, 0.75);
table.captioning(tab, caption);
table.centering(tab);
table.row_sep(tab, 1.1);
table.open_tabular_env(tab, M);
table.panel(tab, "A", M);
table.write_row(tab, '$\E\left[r_{n,t}^{nb,ex}\right]$', rp_NB_mean_to_print);
table.write_row(tab, '$\sigma\hspace{-0.02cm}\left(r_{n,t}^{b,ex}\right)$', rp_NB_vol_to_print);
table.write_row(tab, ['$\E\left[r_{n,t}^{nb,ex}\right]/' ...
    '\sigma\hspace{-0.02cm}\left(r_{n,t}^{nb,ex}\right)$'], sp_NB_to_print);
table.panel(tab, "B", M);
table.write_row(tab, '$\E\left[r_{n,t}^{zce,ex}\right]$', rp_ZCE_mean_to_print);
table.write_row(tab, '$\sigma\hspace{-0.02cm}\left(r_{n,t}^{zce,ex}\right)$', rp_ZCE_vol_to_print);
table.write_row(tab, ['$\E\left[r_{n,t}^{zce,ex}\right]/' ...
    '\sigma\hspace{-0.02cm}\left(r_{n,t}^{zce,ex}\right)$'], sp_ZCE_to_print);
table.write_hline(tab);
table.write_hline(tab);
table.close_tabular_env(tab);
table.close_table_env(tab);

%% COMPILE
compile;


%% FUNCTIONS

function new_string = init(string)
    new_string = strcat(string, "_0");
end

function r = annualised_return(r)
    r = r*12;
end

function std = annualised_std(std)
    std = std*sqrt(12);
end

function compile()
    old_path = getenv("PATH");
    setenv('PATH', '/usr/local/texlive/2022/bin/universal-darwin/');
    system('pdflatex Main_RubenFernandez.tex');
    setenv("PATH", old_path);
    setenv('PATH', '/usr/bin/');
    system('open Main_RubenFernandez.pdf');
end

function infl = inflation_prompt()
    prompt = 'Wanna print inflation?: (Y/n) ';
    inflation = input(prompt, 's');
    inflation_control=0;
    if contains('yes', lower(inflation))
        inflation_control=1;
    else
        inflation_control=0;
    end
    infl = inflation_control;
end
