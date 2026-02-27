% Transient Stability for SG
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
%% Wrapper function Normal operation
function dxdt = f_af(t,x,SG)
global s 
dxdt = [];
for i=1:s+1
    dxdt = [dxdt; f_SG(t,x,SG,i)];
end


    function dxdt = f_SG(t,x,SG,i)
        global Y
        ws = 2*pi*60;
        Pm1 = [2.0802 SG(3,:)];
        Pm = Pm1(i);
        V = [1; SG(4,:).'];
        Kd1 = [0 SG(6,:)];
        Kd = Kd1(i);
        H1 = [1e14 SG(5,:)];
        H = H1(i);

        % Define states
        del = x(2*(1:s+1)-1,1);
        w = x(2*i,1);

        % Define Pe as function of del
        Pe = 0;
        for j = 1:s+1
            Pe = Pe+V(i)*abs(Y(i,j))*V(j)*cos(angle(Y(i,j))+del(j)-del(i));
        end

        % Pe = V(1+i)*abs(Y(1+i,:))*(V.*cos(angle(Y(:,1+i))+del-del(1+i)));

        dxdt(1,1) = ws*(w-1);
        dxdt(2,1) = (Pm-Pe-Kd*(w-1))/(2*H);
    end
end