% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Find quaternion from roll, pitch, yaw
%  This files takes Roll/Pitch/Yaw Euler angles as inputs
% and transforms into the quaternion representation.
% Representative of Euler angle sequence (1,2,3) according to
% "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
% by James Diebel, Stanford University, 20 October 2006
%--------------------------------------------------------------------------
function quat_output = rpy2quat(quat_input)
%-------------------------------------------------------
% System states
roll  = quat_input(1);  % Roll input [rad]
pitch = quat_input(2);  % Pitch input [rad]
yaw   = quat_input(3);  % Yaw input [rad]
%-------------------------------------------------------
phi   = roll / 2;
theta = pitch / 2;
psi   = yaw / 2;
%
cphi   = cos(phi);
sphi   = sin(phi);
ctheta = cos(theta);
stheta = sin(theta);
cpsi   = cos(psi);
spsi   = sin(psi);
%
% This is the (1,2,3) angle sequence computation:
q0 = cphi * ctheta * cpsi + sphi * stheta * spsi;
q1 = sphi * ctheta * cpsi - cphi * stheta * spsi;
q2 = cphi * stheta * cpsi + sphi * ctheta * spsi;
q3 = cphi * ctheta * spsi - sphi * stheta * cpsi;
%-------------------------------------------------------	
if (q0 < 0.0)       % q0 always positive for uniqueness of the quaternion
    q0 = -q0;
	q1 = -q1;
	q2 = -q2;
	q3 = -q3;
end
%-------------------------------------------------------
% Outputs are the four entries of the desired normalized attitude quaternion
quat_output(1,1) = q0;
quat_output(2,1) = q1;
quat_output(3,1) = q2;
quat_output(4,1) = q3;
%--------------------------------------------------------------------------
% end of file