% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Compute rotation matrix from unit quaternion
% Generate the rotation matrix from the unit quaternion
% This is the rotation matrix R_IB, from Inertial axes to Body axes
%--------------------------------------------------------------------------
function rot_output = quat2rot(quat_input)
%-------------------------------------------------------
% System states
q0  = quat_input(1);  % Quaternion input
q1  = quat_input(2);  % Quaternion input
q2  = quat_input(3);  % Quaternion input
q3  = quat_input(4);  % Quaternion input
%-------------------------------------------------------
if (q0 < 0.0)       % q0 always positive for uniqueness of the quaternion
    q0 = -q0;
    q1 = -q1;
	q2 = -q2;
	q3 = -q3;
end
%-------------------------------------------------------
% Output is the rotation matrix R_IB from Inertial axes to Body axes
rot_output = [q0^2+q1^2-q2^2-q3^2 2*(q1*q2+q0*q3) 2*(q1*q3-q0*q2); ...
              2*(q1*q2-q0*q3) q0^2-q1^2+q2^2-q3^2 2*(q2*q3+q0*q1); ...
              2*(q1*q3+q0*q2) 2*(q2*q3-q0*q1) q0^2-q1^2-q2^2+q3^2];
%--------------------------------------------------------------------------
% end of file