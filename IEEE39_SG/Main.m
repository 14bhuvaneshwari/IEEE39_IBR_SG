% Transient Stability for SG
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
% line_no=[3,4,6,7,8,9,10,11,12,13,14,15,17,18,22,23,24,25,27,28,29,30,34,35,37,39,41,42,43,44];
% overall=length(line_no);
% for value=1:1:overall
%   lin=line_no(value);
%   for k=1:1:19

    global s Y 
    %% Dynamic equations
    Ts = 1/60;
    tf = 1;
    sol1 = ode113(@(t,x) f(t,x,SG), [0;tf], x0); % before fault
    tsim=10;
    t1 = 0:Ts:tf-Ts;
    x1 = deval(sol1,t1)';

    %% Create Ybus (during fault network)
    global Y_df
    fault_line =11;%lin; 
    x = 0.5; % fault location 0.05*k
    
    Ybus_df = Ybus;
    i = Line(2,fault_line);
    j = Line(3,fault_line);
    Ybus_df(i,i) = Ybus_df(i,i)+(1-x)*(yline(fault_line)/x-ysh(fault_line));
    Ybus_df(j,j) = Ybus_df(j,j)+x*(yline(fault_line)/(1-x)-ysh(fault_line));
    Ybus_df(i,j) = Ybus_df(i,j)+yline(fault_line);
    Ybus_df(j,i) = Ybus_df(j,i)+yline(fault_line);
    
    % Kron's reduction
    Ybb = Ybus_df(keep,keep);
    Ybr = Ybus_df(keep,remove);
    Yrb = Ybus_df(remove,keep);
    Yrr = Ybus_df(remove,remove);
    Y_df = Ybb-Ybr*(Yrr\Yrb);
    
    %% Create Ybus (after fault network) where 4th line removed
    global Y_af
    Ybus_af = Ybus;
    % fault_line = 10; % line to remove
    i = Line(2, fault_line);
    j = Line(3, fault_line);
    %subtract
    Ybus_af(i,i) = Ybus_af(i,i) - (yline(fault_line) - ysh(fault_line));
    Ybus_af(j,j) = Ybus_af(j,j) - (yline(fault_line) - ysh(fault_line));
    Ybus_af(i,j) = Ybus_af(i,j) + yline(fault_line);
    Ybus_af(j,i) = Ybus_af(j,i) + yline(fault_line);
    
    % Kron's reduction
    Ybb = Ybus_af(keep,keep);
    Ybr = Ybus_af(keep,remove);
    Yrb = Ybus_af(remove,keep);
    Yrr = Ybus_af(remove,remove);
    Y_af = Ybb-Ybr*(Yrr\Yrb);
    % tcl=0.61;
    % tcl_step = Ts; tcl_max = 2;
    % tcr = 0;
    % for tcl = 2*Ts:tcl_step:tcl_max
        % fault clearing time
        tcl=0.5;
        % During fault dynamics
        sol2 = ode113(@(t,x) f_df(t,x,SG), [tf;tf+tcl], x1(end,:)); % during fault
        t2 = tf:Ts:tf+tcl-Ts;
        x2 = deval(sol2,t2)';
        % %% Data storage
        % del=[0,x2(5,2*(1:s)-1)]';%start from 5 th point
        % v_store{lin+1,k+1}=abs(V.*exp(1j*del));
        % vstore_angle{lin+1,k+1}=angle(V.*exp(1j*del));
        % Vdata=V.*exp(1j*del);
        % i_store{lin+1,k+1}= abs(Y_df*Vdata);
        % istore_angle{lin+1,k+1}=angle(Y_df*Vdata);
        % w_store{lin+1,k+1}=[0,x2(5,2*(1:s))]';
        % v_big=zeros(10,24);
        % v_big_angle=zeros(10,24);
        % i_big=zeros(10,24);
        % i_big_angle=zeros(10,24);
        % w_big=zeros(10,24);
        % for i=5:5:120
        %     del1=[0,x2(i,2*(1:s)-1)]';
        %     w1=[1,x2(i,2*(1:s))]';
        %     Vdata1=V.*exp(1j*del1);
        %     ke=i/5;
        %     v_big(:,ke)=abs(V.*exp(1j*del1));
        %     v_big_angle(:,ke)=angle(V.*exp(1j*del1));
        %     i_big(:,ke) = abs(Y_df*Vdata1);
        %     i_big_angle(:,ke)=angle(Y_df*Vdata1);
        %     w_big(:,ke)=w1;
        % end
        % w_obj{lin+1,k+1}=w_big;
        % v_obj_mag{lin+1,k+1}=v_big;
        % v_obj_angle{lin+1,k+1}=v_big_angle;
        % i_obj_mag{lin+1,k+1}=i_big;
        % i_obj_angle{lin+1,k+1}=i_big_angle;

        % Post fault dynamics
        sol3 = ode113(@(t,x) f_af(t,x,SG), [tf+tcl;tsim], x2(end,:)); % post fault
        t3 = tf+tcl:Ts:tsim-Ts;
        x3 = deval(sol3,t3)';
    
        % Plotting
        t = [t1,t2,t3]';
        x = [x1;x2;x3]-x0';
        % if any(max(x3(:,2*(1:s)-1)) > 4)  % Or any(abs(x3(2:2:end)) > 4)
        %     tcr = tcl - tcl_step+Ts;   % Previous was stable
        %     break;
        % end
   %  end
   %   % CCT_table_test(lin+1,k+1)=tcr;
   % end
% end
    subplot(211);plot(t,x(:,2*(1:s+1)-1),'.','LineWidth',2);xlabel('t (s)');ylabel('\Delta\delta');
    subplot(212);plot(t,x(:,2*(1:s+1)),'.','LineWidth',2);xlabel('t (s)');ylabel('\Delta\omega');
