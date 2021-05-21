% read the validation figures and perform the validation
close all

mm = importdata('mean.txt');
mf = importdata('fluc.txt');
cm = importdata('../validation/supersonic-channel/Mach-1.5/data/present.mean');
cf = importdata('../validation/supersonic-channel/Mach-1.5/data/present.fluc');

nmm = mm;
nmf = mf;
ncm = cm;
ncf = cf;


set(groot,'defaultLineLineWidth',1.0);
set(groot,'defaultLineMarkerSize',5.0);
set(groot,'defaultAxesTickLabelInterpreter','latex');

sk = 2;

%% PLOT MEAN

% -- density, temperature and viscosity
fig = figure();
plot(nmm(:,1),nmm(:,2)); hold on;
plot(nmm(:,1),nmm(:,12)); hold on;
plot(nmm(:,1),nmm(:,14)*3000); hold on;
plot(ncm(1:sk:end,1),ncm(1:sk:end,2),'o');
plot(ncm(1:sk:end,1),ncm(1:sk:end,12),'s');
plot(ncm(1:sk:end,1),ncm(1:sk:end,14)*3000,'d');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{\rho}$','Present code, $\overline{T}$','Present code, $\overline{\mu}$',...
    'Coleman, $\overline{\rho}$','Coleman, $\overline{T}$','Coleman, $\overline{\mu}$'},'Interpreter','latex','location','east')
legend boxoff
% print(fig,'prop.eps','-depsc')

% -- pressure
fig = figure();
plot(nmm(:,1),nmm(:,13)); hold on;
plot(ncm(1:sk:end,1),ncm(1:sk:end,13),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{p}$','Coleman, $\overline{p}$'},'Interpreter','latex','location','east')
legend boxoff
% print(fig,'press.eps','-depsc')

% -- velocity reynolds averaged
fig = figure();
plot(nmm(:,1),nmm(:,8)); hold on;
plot(ncm(1:sk:end,1),ncm(1:sk:end,8),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{u}$','Coleman, $\overline{u}$'},'Interpreter','latex','location','east')
legend boxoff
% print(fig,'velrey.eps','-depsc')

% % -- velocity favre averaged
% fig = figure();
% plot(nmm(:,1),nmm(:,5)); hold on;
% plot(ncm(:,1),ncm(:,10),'o');

%% PLOT FLUC

% -- fluctuations favre averaged
% fig = figure();
% plot(nmf(:,1),nmf(:,5)); hold on
% plot(nmf(:,1),nmf(:,3));
% plot(nmf(:,1),nmf(:,4));
% plot(ncf(1:sk:end,1),ncf(1:sk:end,2),'o');
% plot(ncf(1:sk:end,1),ncf(1:sk:end,3),'s');
% plot(ncf(1:sk:end,1),ncf(1:sk:end,4),'d');
% xlabel('$y/h$','interpreter','latex');
% legend({'Present code, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Present code, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Present code, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$',...
%     'Coleman, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Coleman, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Coleman, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$'},'Interpreter','latex','location','northeast')
% legend boxoff
% print(fig,'ChannelMach1.5Re3000/flucfavre.eps','-depsc')
% 
% -- fluctuations reynolds averaged
fig = figure();
plot(nmf(:,1),nmf(:,6)); hold on
plot(nmf(:,1),nmf(:,7));
plot(nmf(:,1),nmf(:,8));
plot(ncf(1:sk:end,1),ncf(1:sk:end,6),'o');
plot(ncf(1:sk:end,1),ncf(1:sk:end,7),'s');
plot(ncf(1:sk:end,1),ncf(1:sk:end,8),'d');xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Present code, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Present code, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$',...
    'Coleman, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Coleman, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Coleman, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$'},'Interpreter','latex','location','northeast')
legend boxoff
% print(fig,'flucrey.eps','-depsc')

% -- temperature fluctuations
fig = figure();
plot(nmf(:,1),nmf(:,12)); hold on
plot(ncf(1:sk:end,1),ncf(1:sk:end,12),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{T^{\prime} T^{\prime}}$','Coleman, $\overline{T^{\prime} T^{\prime}}$'},'Interpreter','latex','location','northeast')
legend boxoff
% print(fig,'fluctemp.eps','-depsc')

















