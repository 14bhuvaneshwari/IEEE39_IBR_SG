% Transient Stability for SG
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
%% Wrapper function During fault
function dxdt = f_df(t,x,SG)
global s
dxdt = [];
for i=1:s+1
    dxdt = [dxdt; f_SG_df(t,x,SG,i)];
end

    function dxdt = f_SG_df(t,x,SG,i)
        global Y_df
        ws = 2*pi*60;
        Pm1 = [2.0802 SG(3,:)];
        Pm = Pm1(i);
        V = [1; SG(4,:).'];
        Kd1 = [0 SG(6,:)];
        Kd = Kd1(i);
        H1 = [1e14 SG(5,:)];
        H = H1(i);
        % Define states
        del =x(2*(1:s+1)-1,1);
        w =  x(2*i,1);

        % v_ref=abs(V(i)*exp(1j*del(i)))
        % Define Pe as function of del
        Pe = 0;
        % for j = 1:s+1
        %     i_total=abs(Y_df(i,j))*V(j)*cos(angle(Y_df(i,j))+del(j)-del(i))
        % end
        % v_ref=abs(V(i)*exp(1j*del(i)))
        % i_max=sqrt(2)*1.2*(Pm/(sqrt(3)*1));
        % if 0.5<v_ref && v_ref<0.9
        %     Ks=(0.9-0.5)/(0-(-i_max));
        %     iqref1=Ks*(v_ref-1);
        %     idref1=sqrt(i_max^2-iqref1^2);
        % elseif v_ref<0.5 
        %     iqref1=-i_max;
        %     idref1=0;
        % elseif v_ref>1.1 && v_ref<1.2
        %     Ks=(1.2-1.1)/(i_max-0);
        %     iqref1=Ks*(v_ref-1);
        %     idref1=sqrt(i_max^2-iqref1^2);
        % elseif v_ref>1.2
        %     iqref1=i_max;
        %     idref1=0;
        %     m=idref1+1j*iqref1
        % end
        % Ibus=Y_df *(V.*exp(del));         
        % Ii=Ibus(i)
        % Pe = Pe+V(i)*i_total;
        Pe = 0;
        for j = 1:s+1
            Pe = Pe+V(i)*abs(Y_df(i,j))*V(j)*cos(angle(Y_df(i,j))+del(j)-del(i));
        end
        dxdt(1,1) = ws*(w-1);
        dxdt(2,1) = (Pm-Pe-Kd*(w-1))/(2*H);
    end
end