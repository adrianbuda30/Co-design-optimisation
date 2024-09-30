%--------------------------------------------------------------------------
%% Multirotor Dynamic Simulation Environment for the Testing of GNC Algorithms
% This simulation is part of the Master's thesis titled 
% "A Modular Simulation Environment for the Improved Dynamic Simulation of
% Multirotor Unmanned Aerial Vehicles",
% submitted to the University of Illinois at Urbana-Champaign in May 2016.
% The thesis itself can be found here: http://hdl.handle.net/2142/90660
% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Initialization routine for the Simulink simulation file Sim_Multi.slx
%--------------------------------------------------------------------------
clear all;close all;clc
%--------------------------------------------------------------------------
t_control = 1/500;          % Controller/state estimation frequency 500Hz [s]
t_sim = t_control/2;        % Simulation frequency 1000Hz [s]
sim_time = 10.0;            % Duration of simulation [sec]
r2d = 180/pi;               % Conversion factor radians to degrees [deg/rad]
d2r = pi/180;               % Conversion factor degrees to radians [rad/deg]
g = 9.80665;                % Acceleration of gravity [m/s^2]
rpm2radpersec = 2*pi/60;    % Conversion factor RPM to rad/sec [rad/(s*rpm)]
rho = 1.225;                % Density of air [kg/m^3]
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Quad specific configuration parameters                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The coordinate system used in this model is defined as follows:
% X: pointing forward, Roll axis
% Y: pointing to the right, Pitch axis
% Z: pointing down, Yaw axis
% To define the geometry of the airframe, the following convention is used:
% - the base airframe is assumed to be somewhat symmetric
% - the CoG for the base airframe coincides with the geometric center in X/Y plane (but is shifted along Z axis)
% - the motor arm length in X/Y plane is defined as the distance to the geometric center of the base airframe
% - motor thrust is generated at the prop, which is offset a distance (along Z) from the geometric center
% - motor thrust vector may be misaligned from purely Z direction, so a set of rotation angles is given for each motor
% - real position of CoG is given as a position vector from geometric center
% - everything is expressed in body coordinate system described above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Noise_flag = 0;                 % Set to 1 for sensor/estimation noise, to 0 for no noise
Coriolis_correction = 1;        % Set to 1 to cancel omega x (J*omega) term in control law, 0 if not
Dyn_thrust = 1;                 % Set to 1 to engage dynamic thrust effects, to 0 to disengage
Blade_flapping = 1;             % Set to 1 to engage blade flapping effects, to 0 to disengage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
Att_init = [0;0;0] * d2r;       % Initial airframe attitude in Euler angles Roll/Pitch/Yaw for 6DOF block [deg]
omega_init = [0;0;0];           % Initial angular rates in Body frame for 6DOF block [rad/s]
Vb_init = [0;0;0];              % Initial velocity in Body frame for 6DOF block [m/s]
Xi_init = [0;0;0];              % Initial position in Inertial frame for 6DOF block [m]
rpm_init = 3104.5025852;        % Initial motor speed [rpm]
q_init = rpy2quat(Att_init);    % Compute t, mx_he initial attitude quaternion
R_IB_init = rpy2rot(Att_init);  % Compute the initial rotation matrix R_IB
Vi_init = R_IB_init' * Vb_init; % Compute the initial velocity in Inertial axes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for aerodynamic drag computation
a = 0.060;                      % Surface area of airframe visible along the Body x axis [m^2]
b = 0.060;                      % Surface area of airframe visible along the Body y axis [m^2]
c = 0.060;                      % Surface area of airframe visible along the Body z axis [m^2]
C_D = 0.40;                     % Drag coefficient [dimensionless]
Surface_params = [a;b;c];       % Combine the surface area parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for dynamic thrust computation
v_h = 4.0;                      % Induced aerodynamic velocity at hover [m/s]
kappa = 1.00;                   % Value for induced power factor in hover (chosen to make transition from different states continuous)
k1 = -1.125;                    % Empirical values taken from Hoffmann_2011
k2 = -1.372;                    % Empirical values taken from Hoffmann_2011
k3 = -1.718;                    % Empirical values taken from Hoffmann_2011
k4 = -0.655;                    % Empirical values taken from Hoffmann_2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for blade flapping computation
k_a1s = 1.5 / 4.0;              % Gain in linear relationship between wind velocity and flapping angle [deg/(m/s)]
k_beta = 0.23;                  % Stiffness of the propeller blades [Nm/rad]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real airframe data
ab
mass_real = 0.550;              % Complete airframe mass [kg]
Jxx = 0.003960;                 %
Jxy = 0;                        %
Jxz = 0;                        %
Jyx = 0;                        %
Jyy = 0.003845;                 % Moment of inertia for multirotor w.r.t center of gravity [kg*m^2]
Jyz = 0;                        %
Jzx = 0;                        %
Jzy = 0;                        %
Jzz = 0.007350;                 %
J_real = [Jxx Jxy Jxz;Jyx Jyy Jyz;Jzx Jzy Jzz]; % Moment of inertia matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MotorMatrix_real holds all the information about the actual performance
% of each actuator. The data is arranged as one row per motor/prop combo:
% 1      : Motor arm angle measured clockwise (looking from above) from the positive X axis (forward direction) [deg]
% 2      : Distance of prop/motor in X/Y plane from the geometric center of the airframe [m]
% 3      : Distance of prop/motor in Z direction from the geometric center of the airframe [m]
% 4      : Direction of prop rotation: -1 for CW, +1 for CCW [unitless]
% 5      : Control effectiveness of the actuator (nominally 1.0)
% 6      : First-order motor transfer function time constant [sec]
% 7..8   : Two coefficients that describe the RPM to thrust [N] transfer function for static conditions [a1 a2]
%          Thrust = a1 * RPM + a2 * RPM^2
% 9..10  : Two coefficients that describe the RPM to torque [Nm] transfer function for static conditions [b1 b2]
%          Torque = b1 * RPM + b2 * RPM^2
% 11     : Minimum RPM value of the actuator
% 12     : Maximum RPM value of the actuator
% 13..15 : Rotations of thrust vector around Body-fixed axes away from nominal direction [deg]
%          Nominal direction of thrust vector is [0;0;-1]
%          Rotation order from body to motor axes is Yaw/Pitch/Roll (Euler angle sequence (1,2,3))
% 16     : Propeller diameter [m]
% 17     : Propeller mass [kg]
MotorMatrix_real = [045 , 0.170 , -0.028 , -1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                    135 , 0.170 , -0.028 , +1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                    225 , 0.170 , -0.028 , -1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                    315 , 0.170 , -0.028 , +1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nominal airframe data (without disturbance/uncertainties)
CoG_nominal = [0;0;0.001];      % Location of center of gravity w.r.t. geometric center (in Body axes) [m]
mass_nominal = 0.550;           % Complete airframe mass [kg]
Jxx = 0.003960;                 %
Jxy = 0;                        %
Jxz = 0;                        %
Jyx = 0;                        %
Jyy = 0.003845;                 % Moment of inertia for multirotor w.r.t center of gravity [kg*m^2]
Jyz = 0;                        %
Jzx = 0;                        %
Jzy = 0;                        %
Jzz = 0.007350;                 %
J_nominal = [Jxx Jxy Jxz;Jyx Jyy Jyz;Jzx Jzy Jzz]; % Moment of inertia matrix
k_torque = 3/200;               % Factor how thrust and induced moment are related (from motor test data)
thrust_reserve = 0.10;          % Limit total commanded thrust to 10...90% of max thrust to still allow maneuverability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MotorMatrix_nominal holds the nominal information that the controller
% assumes about each actuator. Again, the data is arranged as one row per
% motor/prop combo:
% 1      : Motor arm angle measured clockwise (looking from above) from the positive X axis (forward direction) [deg]
% 2      : Distance of prop/motor in X/Y plane from the geometric center of the airframe [m]
% 3      : Distance of prop/motor in Z direction from the geometric center of the airframe [m]
% 4      : Direction of prop rotation: -1 for CW, +1 for CCW [unitless]
% 5      : Control effectiveness of the actuator (nominally 1.0)
% 6      : First-order motor transfer function time constant [sec]
% 7..8   : Two coefficients that describe the RPM to thrust [N] transfer function for static conditions [a1 a2]
%          Thrust = a1 * RPM + a2 * RPM^2
% 9..10  : Two coefficients that describe the RPM to torque [Nm] transfer function for static conditions [b1 b2]
%          Torque = b1 * RPM + b2 * RPM^2
% 11     : Minimum RPM value of the actuator
% 12     : Maximum RPM value of the actuator
% 13..15 : Rotations of thrust vector around Body-fixed axes away from nominal direction [deg]
%          Nominal direction of thrust vector is [0;0;-1]
%          Rotation order from body to motor axes is Yaw/Pitch/Roll (Euler angle sequence (1,2,3))
% 16     : Propeller diameter [m]
% 17     : Propeller mass [kg]
MotorMatrix_nominal = [045 , 0.170 , -0.028 , -1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                       135 , 0.170 , -0.028 , +1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                       225 , 0.170 , -0.028 , -1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000; ...
                       315 , 0.170 , -0.028 , +1 , 1.0 , 0.010 , [9.6820 , 0.010872]*1e-5 , [1.4504 , 0.0016312]*1e-6 , 0000 , 6000 , [0.0 , 0.0 , 0.0] , 8*2.54/100 , 11/1000];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Radio scaling factor                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the scaling factor to turn radio throttle command into a thrust value
max_thrust = 0;
min_thrust = 0;
for i = 1:size(MotorMatrix_nominal,1)
    max_thrust = max_thrust + (MotorMatrix_nominal(i,7)*MotorMatrix_nominal(i,12)+MotorMatrix_nominal(i,8)*MotorMatrix_nominal(i,12)^2);
    min_thrust = min_thrust + (MotorMatrix_nominal(i,7)*MotorMatrix_nominal(i,11)+MotorMatrix_nominal(i,8)*MotorMatrix_nominal(i,11)^2);
end
thrust_gain = (max_thrust - min_thrust) / 1000;
thrust_offset = min_thrust;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                   Compute the motor mapping matrix                      %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % This matrix describes the mapping from body torques to motor thrusts.
for i = 1:size(MotorMatrix_nominal,1)
    % The vector from nominal center of gravity to nominal location of each propeller
    prop_vector(i,1:3) = [abs(MotorMatrix_nominal(i,2)) * [cos(MotorMatrix_nominal(i,1)*d2r);sin(MotorMatrix_nominal(i,1)*d2r)];MotorMatrix_nominal(i,3)] - CoG_nominal;
    % The nominal direction in which each motor generates thrust force
    thrust_direction(i,1:3) = -[cos(MotorMatrix_nominal(i,13)*d2r)*sin(MotorMatrix_nominal(i,14)*d2r)*cos(MotorMatrix_nominal(i,15)*d2r)+sin(MotorMatrix_nominal(i,13)*d2r)*sin(MotorMatrix_nominal(i,15)*d2r); ...
                              cos(MotorMatrix_nominal(i,13)*d2r)*sin(MotorMatrix_nominal(i,14)*d2r)*sin(MotorMatrix_nominal(i,15)*d2r)-sin(MotorMatrix_nominal(i,13)*d2r)*cos(MotorMatrix_nominal(i,15)*d2r); ...
                              cos(MotorMatrix_nominal(i,13)*d2r)*cos(MotorMatrix_nominal(i,14)*d2r)];
    % The B matrix relates motor thrust to body torques and thrust
    % Each column describes one motor/prop. Entries by row 1-4:
    % contributions to roll/pitch/yaw torques and thrust, respectively    
	B(1:3,i) = cross(prop_vector(i,:)',thrust_direction(i,:)') + k_torque*thrust_direction(i,:)'*MotorMatrix_nominal(i,4);
    B(4,i) = 1.0;    
end
% For any airframe with a number of actuators other than four, the B matrix
% is nonsquare and therefore taking the regular inverse will not work.
% Instead we use the Moore-Penrose pseudoinverse for matrices with linearly 
% independent rows:
MotorMap = B' * inv(B * B');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Tuning parameters for the controllers                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Controller settings
Att_cmd_init = [0;0;0] * d2r;           % Initial attitude command for quaternion strapdown equation initialization in control law
MaxRate_cmd = [360;360;120] * d2r;      % Maximum rate command at full stick deflection [rad/s]
MaxAtt_cmd = 50 * d2r;                  % Max angle command at full stick deflection [rad]
Ref_bw = 10.0;                          % Reference system bandwidth for rate command generation in attitude control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gains for simple PI controller on rates
Kp_1  = 15*[1;1;1.5];
Ki_1  = 10*[1;1;1.5];
Kff_1 = 0.7*[1;1;0.95];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Execute the simulation                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sim('Sim_Multi');
%--------------------------------------------------------------------------
