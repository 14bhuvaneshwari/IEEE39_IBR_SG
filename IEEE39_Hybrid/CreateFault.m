% Transient Stability for Hybrid system
% Authors: Pankaj Dilip Achlerkar, Bhuvaneshwari Shaktawat
% Affiliation: IIT Bhubaneswar, Department of Electrical Engineering
% Date: February 22, 2026
function [Ybus_df,Ybus_af] = CreateFault(fault_line,x)
global Line Ybus_L SL_o PV_o CC_o m
i = Line(2,fault_line);
j = Line(3,fault_line);
yline = 1/(Line(4,fault_line)+1j*Line(5,fault_line));
ysh = 0.5j*Line(6,fault_line);

%% During fault
Ybus_df = Ybus_L; % Initialize
Ybus_df(i,i) = Ybus_df(i,i)+(1-x)*(yline/x-ysh);
Ybus_df(j,j) = Ybus_df(j,j)+x*(yline/(1-x)-ysh);
Ybus_df(i,j) = Ybus_df(i,j)+yline;
Ybus_df(j,i) = Ybus_df(j,i)+yline;

% Kron's reduction
keep1 = [SL_o;PV_o;CC_o];
remove1 = setdiff((1:m).',[SL_o;PV_o;CC_o]);
Y1_df = Ybus_df(keep1,keep1);
Y2_df = Ybus_df(keep1,remove1);
Y3_df = Ybus_df(remove1,keep1);
Y4_df = Ybus_df(remove1,remove1);
Ybus_df = Y1_df-Y2_df*(Y4_df\Y3_df); % size (1+s+b)

%% After fault line out
Ybus_af = Ybus_L; % Initialize
Ybus_af(i,i) = Ybus_af(i,i)-yline-ysh;
Ybus_af(j,j) = Ybus_af(j,j)-yline-ysh;
Ybus_af(i,j) = Ybus_af(i,j)+yline;
Ybus_af(j,i) = Ybus_af(j,i)+yline;

% Kron's reduction
Y1_af = Ybus_af(keep1,keep1);
Y2_af = Ybus_af(keep1,remove1);
Y3_af = Ybus_af(remove1,keep1);
Y4_af= Ybus_af(remove1,remove1);
Ybus_af = Y1_af-Y2_af*(Y4_af\Y3_af); % size (1+s+b)