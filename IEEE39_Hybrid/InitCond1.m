% Transient Stability for Hybrid system
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
clc;clear;close;

global s b m SG IBR Line Load SL_o PV_o CC_o SL PV CC Ybus Ybus_L Vbase Ibase ws Icc

% Import data
SG = xlsread('data_IEEE39.xlsx','SG');
Line = xlsread('data_IEEE39.xlsx','Line');
Load = xlsread('data_IEEE39.xlsx','Load');
IBR =  xlsread('data_IEEE39.xlsx','Ibrs');

% Numbers
s = length(SG(1,:)); % no. of SGs (excluding slack)
b = length(IBR(1,:)); % no. of IBRs
n = length(Line(1,:)); % no. of lines
m = max(max(Line(2:3,:))); % no. of original buses
p = length(Load(1,:)); % no. of loads

% System data
Sbase = 100e6;
Vbase = 345e3;
Ibase = Sbase/sqrt(3)/Vbase;
Zbase = Vbase^2/Sbase;
ws = 2*pi*60;

% Type of buses (original numbering)
SL_o = 39; % slack
PV_o = SG(2,:).';
PQ_o = Load(2,:).';
CC_o = IBR(2,:).';

% Ybus (original)
Ybus_o = zeros(m);
yline = 1./(Line(4,:)+1j*Line(5,:));
ysh = 0.5j*Line(6,:);

for i = 1:n % lines
    from = Line(2,i);
    to = Line(3,i);
    Ybus_o(from,from) = Ybus_o(from,from)+yline(i)+ysh(i);
    Ybus_o(to,to) = Ybus_o(to,to)+yline(i)+ysh(i);
    Ybus_o(from,to) = Ybus_o(from,to)-yline(i);
    Ybus_o(to,from) = Ybus_o(to,from)-yline(i);
end
for i = 1:s % SG reactances
    genbus = SG(2,i);
    Ybus_o(genbus,genbus) = Ybus_o(genbus,genbus)-1j/SG(7,i);
end

% Remove zero-injection buses
keep = [SL_o;PV_o;PQ_o;CC_o];
remove = setdiff((1:m).',[SL_o;PV_o;PQ_o;CC_o]); % zero injection buses
Y1 = Ybus_o(keep,keep);
Y2 = Ybus_o(keep,remove);
Y3 = Ybus_o(remove,keep);
Y4 = Ybus_o(remove,remove);
Ybus = Y1-Y2*(Y4\Y3); % size (1+s+p+b)

% New bus numbering
SL = 1;
PV = 1+(1:s).';
PQ = 1+s+(1:p).';
CC = 1+s+p+(1:b).';
PVPQ = [PV;PQ];

% Ybus partitioning for reducing IBR buses
Y1 = Ybus([SL;PV;PQ],[SL;PV;PQ]);
Y2 = Ybus([SL;PV;PQ],CC);
Y3 = Ybus(CC,[SL;PV;PQ]);
Y4 = Ybus(CC,CC);

% Ybus after reducing IBR bus
Yred = Y1-Y2*(Y4\Y3); % size (1+s+p)
G = real(Yred); B = imag(Yred);
Y = abs(Yred); th = angle(Yred);

% Actual to pu conversion for IBR currents
Icc = (IBR(16,:)-1j*IBR(17,:)).'/(sqrt(2)*Ibase); % IBR injected currents (pu)

tic;
% Initialize for load flow
V = [1; SG(4,:).'; ones(p,1)]; % Voltage magnitude of Slack, PV, PQ
del = zeros(1+s+p,1); % Voltage angles

% Specified powers of Slack, PV, PQ
P_sp = [0; SG(3,:).'; -Load(3,:).'];
Q_sp = [0; zeros(s,1); -Load(4,:).'];

err = inf;
iter = 0;
while err>1e-6
    I_CC = Y2*(Y4\Icc);
    ICC = abs(I_CC); phi = angle(I_CC);

    % Calculate powers of Slack, PV, PQ
    P_cal = V.*ICC.*cos(phi-del);
    Q_cal = -V.*ICC.*sin(phi-del);
    for i = 1:1+s+p
        for j = 1:1+s+p
            P_cal(i) = P_cal(i)+V(i)*V(j)*Y(i,j)*cos(th(i,j)+del(j)-del(i));
            Q_cal(i) = Q_cal(i)-V(i)*V(j)*Y(i,j)*sin(th(i,j)+del(j)-del(i));
        end
    end

    % Jacobian calculation: dP/ddel, size (s+p)x(s+p)
    dP_ddel = zeros(s+p);
    for I = 1:(s+p)
        i = PVPQ(I);
        for J = 1:(s+p)
            j = PVPQ(J);
            if i==j
                dP_ddel(I,J) = -V(i)^2*B(i,i)-Q_cal(i);
            else
                dP_ddel(I,J) = -V(i)*V(j)*Y(i,j)*sin(th(i,j)+del(j)-del(i));
            end
        end
    end

    % Jacobian calculation: dP/dV, size (s+p)x(p)
    dP_dV = zeros(s+p,p);
    for I = 1:(s+p)
        i = PVPQ(I); % bus no.
        for J = 1:p
            j = PQ(J);
            if i==j
                dP_dV(I,J) = V(i)*G(i,i)+P_cal(i)/V(i);
            else
                dP_dV(I,J) = V(i)*Y(i,j)*cos(th(i,j)+del(j)-del(i));
            end
        end
    end

    % Jacobian calculation: dQ/ddel, size (p)x(s+p)
    dQ_ddel = zeros(p,s+p);
    for I = 1:p
        i = PQ(I); % bus no.
        for J = 1:(s+p)
            j = PVPQ(J);
            if i==j
                dQ_ddel(I,J) = P_cal(i)-V(i)^2*G(i,i);
            else
                dQ_ddel(I,J) = -V(i)*V(j)*Y(i,j)*cos(th(i,j)+del(j)-del(i));
            end
        end
    end

    % Jacobian calculation: dQ/dV, size (p)x(p)
    dQ_dV = zeros(p);
    for I = 1:p
        i = PQ(I); % bus no.
        for J = 1:p
            j = PQ(J);
            if i==j
                dQ_dV(I,J) = -V(i)*B(i,i)+Q_cal(i)/V(i);
            else
                dQ_dV(I,J) = -V(i)*Y(i,j)*sin(th(i,j)+del(j)-del(i));
            end
        end
    end

    Jac = [dP_ddel, dP_dV; dQ_ddel, dQ_dV]; % Jacobian, size (s+2p)x(s+2p)
    corr = Jac\[P_sp(PVPQ)-P_cal(PVPQ); Q_sp(PQ)-Q_cal(PQ)];
    del(PVPQ) = del(PVPQ)+corr(1:(s+p)); % Update angles of PV,PQ buses
    V(PQ) = V(PQ)+corr(s+p+(1:p)); % Update voltages of PQ buses
    Vcc = Y4\(Icc-Y3*(V.*exp(1j*del))); % Calculate IBR bus voltages
    Icc = abs(Icc).*exp(1j*angle(Vcc)); % Update IBR current angles

    err = max(abs(corr));
    iter = iter+1;
    if iter==50
        fprintf('--- Load Flow Diverged...\n')
        break
    end
end
if iter<50
    fprintf('--- Load flow converged in %d iterations, err is %f, time taken %f s\n',iter,err,toc);
end

% Ybus reduction by removing loads (PQ buses)
yload = transpose(Load(3,:)-1j*Load(4,:))./(V(PQ).^2); % using converged bus voltages
Ybus_L = Ybus_o;
for i = 1:p % Augment load adittances to existing Ybus
    Ybus(PQ(i),PQ(i)) = Ybus(PQ(i),PQ(i))+yload(i);
    Ybus_L(Load(2,i),Load(2,i)) = Ybus_L(Load(2,i),Load(2,i))+yload(i);
end

keep = [SL;PV;CC];
remove = PQ;
Y1 = Ybus(keep,keep);
Y2 = Ybus(keep,remove);
Y3 = Ybus(remove,keep);
Y4 = Ybus(remove,remove);
Ybus = Y1-Y2*(Y4\Y3); % size (1+s+b)

% New bus numbering
SL = 1;
PV = 1+(1:s).';
CC = 1+s+(1:b).';

% Initial states of SG (del,dw), IBR (del,xpll), size (2s+2b)x1
x0 = [[del(PV);angle(Vcc)].'; ones(1,s+b)];
x0 = x0(:);


