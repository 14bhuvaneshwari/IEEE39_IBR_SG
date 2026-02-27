% Transient Stability for Hybrid System
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
function dxdt = f1(t,x,Ybus)
global s b SG IBR SL PV CC Vbase Ibase ws

Icc = exp(1j*x(2*s+2*(1:b)-1,1)).*((IBR(16,:)-1j*IBR(17,:)).')/(sqrt(2)*Ibase); % Current injected by IBRs
V1 = [1; SG(4,:).'.*exp(1j*x(2*(1:s)-1,1))]; % Voltage phasors of Slack, PV
Vcc = Ybus(CC,CC)\(Icc-Ybus(CC,[SL;PV])*V1); % Voltage at all IBR buses

dxdt = zeros(2*s+2*b,1);
for k = 1:s
    dxdt(2*k-1:2*k,1) = f_SG(t,x,k,Vcc,Ybus);
end
for k = 1:b
    dxdt(2*s+(2*k-1:2*k),1) = f_IBR(t,x,k,Vcc,Ybus);
end

    function dxdt = f_SG(t,x,I,Vcc,Ybus) % for I^th SG
        Pm = SG(3,I); % Mechanical power input
        H = SG(5,I); % Inertia constant
        Kd = SG(6,I)*0; % Damping coefficient
        del = [0; x(2*(1:(s+b))-1,1)]; % angle of all buses (Slack, PV, IBR)

        V = [1; SG(4,:).'; abs(Vcc)]; % voltage magnitudes of all buses (Slack, PV, IBR)
        w = x(2*I); % dw of I^th SG

        Pe = 0; % Initialize
        i = 1+I;
        for j = 1:1+s+b
            Pe = Pe+V(i)*V(j)*abs(Ybus(i,j))*cos(angle(Ybus(i,j))+del(j)-del(i));
        end
        dxdt(1,1) = ws*(w-1);
        dxdt(2,1) = (Pm-Pe)/(2*H);
    end

    function dxdt = f_IBR(t,x,I,Vcc,Ybus) % for I^th IBR
        Kp = IBR(8,I)*0;
        Ki = IBR(9,I);
        % vq = imag(Vcc(I)*exp(-1j*x(2*s+2*I-1,1)))*Vbase*sqrt(2/3);
        vq = imag(Vcc(I)*exp(-1j*x(2*s+2*I-1,1)))*Vbase*sqrt(2/3);
        % xpll = x(2*s+2*I,1);
        % dxdt(1,1) = Kp*vq+Ki*xpll;
        % dxdt(2,1) = vq;
        w1 = x(2*s+2*I,1);
        dxdt(1,1) = (w1-1)*ws;
        dxdt(2,1) = (vq*Ki)/(ws);
    end
end