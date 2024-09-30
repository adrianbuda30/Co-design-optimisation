% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Find  roll, pitch, yaw from quaternion
% This files takes a quaternion as input and transforms it 
% into Roll/Pitch/Yaw Euler angles.
% Representative of Euler angle sequence (1,2,3) according to
% "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
% by James Diebel, Stanford University, 20 October 2006
%--------------------------------------------------------------------------
function euler_output = quat2rpy(quat_input)
%-------------------------------------------------------
% System states
q0 = quat_input(1); % Quaternion input 0
q1 = quat_input(2); % Quaternion input 1
q2 = quat_input(3); % Quaternion input 2
q3 = quat_input(4); % Quaternion input 3
%-------------------------------------------------------
% This is the (1,2,3) angle sequence computation:
roll  = atan2(2*q2*q3+2*q0*q1,q3^2-q2^2-q1^2+q0^2);
pitch = -asin(2*q1*q3-2*q0*q2);
yaw   = atan2(2*q1*q2+2*q0*q3,q1^2+q0^2-q3^2-q2^2); 
%-------------------------------------------------------
% Outputs are the three Euler angles Roll, Pitch, Yaw
euler_output(1) = roll;
euler_output(2) = pitch;
euler_output(3) = yaw;
%--------------------------------------------------------------------------
% end of file