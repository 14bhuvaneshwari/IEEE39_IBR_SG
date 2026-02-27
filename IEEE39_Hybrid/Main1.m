% Transient Stability for Hybrid System
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026global Icc
line_no=[3,4,6,7,8,9,10,11,12,13,14,15,17,18,22,23,24,25,27,28,29,30,34,35,37,39,41,42,43,44];
overall=length(line_no);
for value=1:1:overall
  lin=line_no(value);
  for k=1:1:1
    Ts = 1/60; % sampling time
    tf = 1; % fault inception time

    % Before fault
    sol1 = ode15s(@(t,x) f1(t,x,Ybus), [0;tf], x0);
    t1 = 0:Ts:tf-Ts;
    x1 = deval(sol1,t1)';

    tsim = 10; % simulation time
    
    fault_line =lin; 
    x = 0.05*k; % fault location 0.05*k
    [Ybus_df,Ybus_af] = CreateFault(fault_line,x);
    

    % During fault
    % tcl=0.43;
    tcl_step = Ts; tcl_max =2;
    tcr = 0;

    for tcl = 2*Ts:tcl_step:tcl_max
    %tcl=2;
        sol2 = ode15s(@(t,x) f1(t,x,Ybus_df), [tf;tf+tcl], x1(end,:));
        t2 = tf:Ts:tf+tcl-Ts;
        x2 = deval(sol2,t2)';

    %%data collect
    % del=[0,x2(5,2*(1:s)-1)]';%start from 5 th point
    % Icc_ibr = exp(1j*x2(5,2*s+2*(1:b)-1)').*((IBR(16,:)-1j*IBR(17,:)).')/(sqrt(2)*Ibase); % Current injected by IBRs
    % V1 = [1; SG(4,:).'.*exp(1j*x2(5,2*(1:s)-1))']; % Voltage phasors of Slack, PV
    % Vcc = Ybus_df(CC,CC)\(Icc_ibr-Ybus_df(CC,[SL;PV])*V1); % Voltage at all IBR buses
    % V=[V1;Vcc];
    % v_store{lin+1,k+1}=abs(V);
    % vstore_angle{lin+1,k+1}=angle(V);
    % i_store{lin+1,k+1}= abs(Ybus_df*V);
    % istore_angle{lin+1,k+1}=angle(Ybus_df*V);
    % w_store{lin+1,k+1}=[1,x2(5,2*(1:s+b))]';
    % v_big=zeros(10,24);
    % v_big_angle=zeros(10,24);
    % i_big=zeros(10,24);
    % i_big_angle=zeros(10,24);
    % w_big=zeros(10,24);
    % for i=5:5:120
    %     Icc_ibr1 = exp(1j*x2(i,2*s+2*(1:b)-1)').*((IBR(16,:)-1j*IBR(17,:)).')/(sqrt(2)*Ibase); % Current injected by IBRs
    %     V11 = [1; SG(4,:).'.*exp(1j*x2(i,2*(1:s)-1))']; % Voltage phasors of Slack, PV
    %     Vcc1 = Ybus_df(CC,CC)\(Icc_ibr1-Ybus_df(CC,[SL;PV])*V11); % Voltage at all IBR buses
    %     Vdata=[V11;Vcc1];
    %     w1=[1,x2(i,2*(1:s+b))]';
    %     ke=i/5;
    %     v_big(:,ke)=abs(Vdata);
    %     v_big_angle(:,ke)=angle(Vdata);
    %     i_big(:,ke) = abs(Ybus_df*Vdata);
    %     i_big_angle(:,ke)=angle(Ybus_df*Vdata);
    %     w_big(:,ke)=w1;
    % end
    %     w_obj{lin+1,k+1}=w_big;
    %     v_obj_mag{lin+1,k+1}=v_big;
    %     v_obj_angle{lin+1,k+1}=v_big_angle;
    %     i_obj_mag{lin+1,k+1}=i_big;
    %     i_obj_angle{lin+1,k+1}=i_big_angle;

    % After fault
        sol3 = ode15s(@(t,x) f1(t,x,Ybus), [tf+tcl;tsim], x2(end,:));
        t3 = tf+tcl:Ts:tsim-Ts;
        x3 = deval(sol3,t3)';

        % Plotting
        t = [t1,t2,t3].';
        x = [x1;x2;x3];
        if any(max(x3(:,2*(1:s+b)-1)) > 4)  % Or any(abs(x3(2:2:end)) > 4)
            tcr = tcl - tcl_step+Ts;  % Previous was stable
            CCT_table_test(lin+1,k+1)=tcr;
            break;
        end
         % CCT_table_test(lin+1,k+1)=tcr;
   end
    % subplot(211);plot(t,x(:,1:2*(s+b)),linewidth=2); % SG states
    % subplot(212);plot(t,x(:,2*s+(1:2*b)),linewidth=2); % IBR states
  end
end