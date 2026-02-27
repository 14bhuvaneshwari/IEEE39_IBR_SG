% Transient Stability for SG
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
clc;clear all;close all;
global s Y 

SG = xlsread('data_IEEE39.xlsx','SG');
Line = xlsread('data_IEEE39.xlsx','Line');
Load = xlsread('data_IEEE39.xlsx','Load');

% Synchronous generator
Xsg = 1j*SG(7,:);
gen_bus = SG(2,:);
s = length(SG(1,:)); % no. of generators

% Lines
yline = 1./(Line(4,:)+1j*Line(5,:));
ysh = 1j*Line(6,:);
m = max(max(Line(2:3,:))); % no. of original buses
n = length(Line(1,:)); % no. of lines

% Loads
yload = Load(3,:)-1j*Load(4,:);
p = length(Load(1,:)); % no. of loads

% bustype: 0 = PQ, 1 = PV (gen), 2 = Slack
busno = 1:(m+s);
bustype = zeros(m+s,1);
bustype(m+(1:s),1) = 1; % PV
bustype(39,1) = 2; % Slack

%% Create Ybus (original network)
Ybus = zeros(m+s);
for i = 1:n % Add line information to Ybus
    from = Line(2,i);
    to = Line(3,i);
    Ybus(from,from) = Ybus(from,from)+yline(i)+ysh(i);
    Ybus(to,to) = Ybus(to,to)+yline(i)+ysh(i);
    Ybus(from,to) = Ybus(from,to)-yline(i);
    Ybus(to,from) = Ybus(to,from)-yline(i);
end

for i = 1:s % Add generator reactance to Ybus
    to = m+i;
    from = gen_bus(i);
    Ybus(from,from) = Ybus(from,from)+1/Xsg(i);
    Ybus(to,to) = Ybus(to,to)+1/Xsg(i);
    Ybus(from,to) = Ybus(from,to)-1/Xsg(i);
    Ybus(to,from) = Ybus(to,from)-1/Xsg(i);
end

for i = 1:p % Add load to Ybus
    loadbus = Load(2,i);
    Ybus(loadbus,loadbus) = Ybus(loadbus,loadbus)+yload(i);
end

% Kron's reduction
remove = busno(bustype==0);
keep = busno(bustype==1 | bustype==2);
Ybb = Ybus(keep,keep);
Ybr = Ybus(keep,remove);
Yrb = Ybus(remove,keep);
Yrr = Ybus(remove,remove);
Y = Ybb-Ybr*(Yrr\Yrb);

%% Gauss-Siedel Load flow

% Initialize
V = [1; SG(4,:).']; % [slack, gen1, gen2, ...]
del = zeros(1+s,1); % 1 slack bus, s gen buses
Vbus = V.*exp(1j*del);
P = [0; SG(3,:).'];

err = inf;
iter = 0;

% Iterations
while err>=1e-6
    Vbus0 = Vbus;
    for i=2:1+s % for PV (gen) buses
        Q(i) = -imag(conj(Vbus(i))*Y(i,:)*Vbus);
        del(i) = angle(((P(i)-1j*Q(i))/conj(Vbus(i))-Y(i,:)*Vbus)/Y(i,i)+Vbus(i));
        Vbus(i) = abs(Vbus(i))*exp(1j*del(i));
    end
    iter = iter+1;
    err(iter) = max(abs(Vbus0-Vbus));

    if iter>=500
        disp('Load flow diverged')
        break
    end
end

%% Initial state x0 = [del0; w0]
for i = 1:s+1
    x0(2*i-1,1) = angle(Vbus(i)); % del0
    x0(2*i,1) = 1; % w0
end

