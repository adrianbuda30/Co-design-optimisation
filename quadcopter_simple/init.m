clear

%simulation
basic_time_step=0.001;


%VICON
VICON_time_step=0.016;


%IMU
IMU_time_step=0.002;


%initial conditions
omega_0=[0 0 0];
p_0=[0 0 0];
V_0=[0 0 0];
R_0=eye(3);

%initial rotational speed of the rotors
w_0=0*[1 1 1 1];

%gravity
g=9.81;

%inertia and mass
I=0.025*eye(3);
I_inv=inv(I);

%mass of the QC
m=1.5;

%nominal mass of the QC
m0=m*1;


%min and max squared rotor speed in rad/s
w_min=[0 0 0 0];
w_max=[inf inf inf inf].^2;


%length of 1 quadcopter arm
L=0.3; %to be checked!!

%motor coefficients FROM THE UPENN PAPER ON RAM. 
k_F=1; %6.11e-8/0.1047;  %N/(rad/s^2)
k_M=1; %1.5e-9/0.1047;  %Nm/(rad/s^2)


%transformation matrix between w^2 and Thrust/Torques
T_act=[-k_F  -k_F  -k_F  -k_F;
    0 -L*k_F 0 L*k_F;
    L*k_F 0 -L*k_F 0;
    -k_M k_M -k_M k_M];



T_act_inv=inv(T_act);

%gain of the motor controller
k_m=200;



%filter for velocity estimation
wn=2*pi*40;
csi=1;
fil_vel=tf([wn*wn 0], [1 2*csi*wn wn*wn]);
fil_vel=c2d(fil_vel, VICON_time_step, 'tustin');





%ATTITUDE CONTROLLER

KDR = 5; 
KPR = 35;

KDP = 2; 
KPP = 35;

KDY = 5;
KPY = 35;


% KDR = 11; 
% KPR = 30;
% 
% KDP = 11; 
% KPP = 30;
% 
% KDY = 11;
% KPY = 30;


% KDR = 20; 
% KPR = 100;
% 
% KDP = 20; 
% KPP = 100;
% 
% KDY = 20;
% KPY = 100;



%POSITION CONTROLLER
kd=[0.8 0 0;
    0 2 0
    0 0 7];

kp=[0.7 0 0;
    0 1 0;
    0 0 10];

ki=0*[1.45 0 0;
    0 .1 0;
    0 0 2];


%max_roll and max_pitch cmds
max_roll=sin(15*pi/180);
max_pitch=sin(15*pi/180);


%maximal and minimal value of mass
max_m=2;
min_m=1;



%low-pass filt for the adaptation
a=.1;
adapt_tf=tf([a],[1 a]);
adapt_tf=c2d(adapt_tf, VICON_time_step, 'tustin');
% [num_tf, den_tf] = tfdata(adapt_tf);
% 
% num_tf=cell2mat(num_tf);
% den_tf=cell2mat(den_tf);


%for the forgetting factor...
lambda0=1;
k0=1;



%IMU DATA FUSION

%bias in the gyro
omega_bias=0*[0.01;-0.005;0.005];

%initial estimation of orientation
Eul_hat_0=[0;0;0];

%estimation of the gyro bias
b_hat_0=[0;0;0];


%directions of gravity and magnetic field
e_g=[0;0;1];
e_m=[1;0;0];




%weights for gravity and magnetic field meas.
k1=1;
k2=1;

k_P=5;
k_I=10;






% %complimentary filters
% LP_freq=2*pi*10;
% F1=tf([LP_freq^2], [1 2*LP_freq LP_freq^2]);
% F2=(1-F1);
