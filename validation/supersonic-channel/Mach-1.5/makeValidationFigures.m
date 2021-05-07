% read the validation figures and perform the validation
close all

mm = importdata('present-mean');
mf = importdata('present-fluc');
cm = importdata('coleman.mean');
cf = importdata('coleman.fluc');

cm(:,1) = cm(:,1)+1;
cf(:,1) = cf(:,1)+1;

if mod(length(cf(:,1)),2) == 0
    h = length(cf(:,1))/2;
    one = 0;
else
    h = floor(length(cf(:,1))/2);
    one = 1;
end


nmm(:,1) = mm(1:end/2,1);
ncm(:,1) = cm(1:h,1);
nmf(:,1) = mf(1:end/2,1);
ncf(:,1) = cf(1:h,1);

for i = 2:length(mm(1,:))
    nmm(:,i) = (mm(1:end/2,i) + mm(end:-1:end/2+1,i))/2;
    nmf(:,i) = (mf(1:end/2,i) + mf(end:-1:end/2+1,i))/2;
end
for i = 2:length(cm(1,:))
    ncm(:,i) = (cm(1:h,i) + cm(end:-1:h+one+1,i))/2;
    ncf(:,i) = (cf(1:h,i) + cf(end:-1:h+one+1,i))/2;
end

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
plot(ncm(1:sk:end,1),ncm(1:sk:end,4),'s');
plot(ncm(1:sk:end,1),ncm(1:sk:end,8),'d');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{\rho}$','Present code, $\overline{T}$','Present code, $\overline{\mu}$',...
    'Coleman, $\overline{\rho}$','Coleman, $\overline{T}$','Coleman, $\overline{\mu}$'},'Interpreter','latex','location','east')
legend boxoff
print(fig,'prop.eps','-depsc')

% -- pressure
fig = figure();
plot(nmm(:,1),nmm(:,13)); hold on;
plot(ncm(1:sk:end,1),ncm(1:sk:end,3),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{p}$','Coleman, $\overline{p}$'},'Interpreter','latex','location','east')
legend boxoff
print(fig,'press.eps','-depsc')

% -- velocity reynolds averaged
fig = figure();
plot(nmm(:,1),nmm(:,8)); hold on;
plot(ncm(1:sk:end,1),ncm(1:sk:end,5),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{u}$','Coleman, $\overline{u}$'},'Interpreter','latex','location','east')
legend boxoff
print(fig,'velrey.eps','-depsc')

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
plot(ncf(1:sk:end,1),ncf(1:sk:end,9),'o');
plot(ncf(1:sk:end,1),ncf(1:sk:end,10),'s');
plot(ncf(1:sk:end,1),ncf(1:sk:end,8),'d');xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Present code, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Present code, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$',...
    'Coleman, $\widetilde{u^{\prime \prime} u^{\prime \prime}}$','Coleman, $\widetilde{v^{\prime \prime} v^{\prime \prime}}$','Coleman, $\widetilde{w^{\prime \prime} w^{\prime \prime}}$'},'Interpreter','latex','location','northeast')
legend boxoff
print(fig,'flucrey.eps','-depsc')

% -- temperature fluctuations
fig = figure();
plot(nmf(:,1),nmf(:,12)); hold on
plot(ncf(1:sk:end,1),ncf(1:sk:end,12),'o');
xlabel('$y/h$','interpreter','latex');
legend({'Present code, $\overline{T^{\prime} T^{\prime}}$','Coleman, $\overline{T^{\prime} T^{\prime}}$'},'Interpreter','latex','location','northeast')
legend boxoff
print(fig,'fluctemp.eps','-depsc')

















