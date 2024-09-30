% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Compute rotation matrix from unit quaternion
% Generate the unit attitude quaternion from the rotation matrix
% The input is the rotation matrix R_IB from Inertial axes to Body axes
% Algorithm according to
% "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
% by James Diebel, Stanford University, 20 October 2006
%--------------------------------------------------------------------------
function quat_output = rot2quat(rot_input)
%-------------------------------------------------------
% System states
R = rot_input;      % Rotation matrix input
%-------------------------------------------------------
if (R(2,2) > -R(3,3)) && (R(1,1) > -R(2,2)) && (R(1,1) > -R(3,3))
    q0 = 1/2 * sqrt(1+R(1,1)+R(2,2)+R(3,3));
    q1 = 1/2 * (R(2,3)-R(3,2)) / sqrt(1+R(1,1)+R(2,2)+R(3,3));
    q2 = 1/2 * (R(3,1)-R(1,3)) / sqrt(1+R(1,1)+R(2,2)+R(3,3));
    q3 = 1/2 * (R(1,2)-R(2,1)) / sqrt(1+R(1,1)+R(2,2)+R(3,3));
elseif (R(2,2) < -R(3,3)) && (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
	q0 = 1/2 * (R(2,3)-R(3,2)) / sqrt(1+R(1,1)-R(2,2)-R(3,3));
    q1 = 1/2 * sqrt(1+R(1,1)-R(2,2)-R(3,3));
    q2 = 1/2 * (R(1,2)+R(2,1)) / sqrt(1+R(1,1)-R(2,2)-R(3,3));
    q3 = 1/2 * (R(3,1)+R(1,3)) / sqrt(1+R(1,1)-R(2,2)-R(3,3));
elseif (R(2,2) > R(3,3)) && (R(1,1) < R(2,2)) && (R(1,1) < -R(3,3))
    q0 = 1/2 * (R(3,1)-R(1,3)) / sqrt(1-R(1,1)+R(2,2)-R(3,3));
    q1 = 1/2 * (R(1,2)+R(2,1)) / sqrt(1-R(1,1)+R(2,2)-R(3,3));
    q2 = 1/2 * sqrt(1-R(1,1)+R(2,2)-R(3,3));
    q3 = 1/2 * (R(2,3)+R(3,2)) / sqrt(1-R(1,1)+R(2,2)-R(3,3));
elseif (R(2,2) < R(3,3)) && (R(1,1) < -R(2,2)) && (R(1,1) < R(3,3))
    q0 = 1/2 * (R(1,2)-R(2,1)) / sqrt(1-R(1,1)-R(2,2)+R(3,3));
    q1 = 1/2 * (R(3,1)+R(1,3)) / sqrt(1-R(1,1)-R(2,2)+R(3,3));
    q2 = 1/2 * (R(2,3)+R(3,2)) / sqrt(1-R(1,1)-R(2,2)+R(3,3));
    q3 = 1/2 * sqrt(1-R(1,1)-R(2,2)+R(3,3));
else
    error('Coding error');
end
%-------------------------------------------------------
if (q0 < 0.0)       % q0 always positive for uniqueness of the quaternion
	q0 = -q0;
    q1 = -q1;
	q2 = -q2;
	q3 = -q3;
end
%-------------------------------------------------------
% Output is the unit quaternion
quat_output = [q0;q1;q2;q3];
%--------------------------------------------------------------------------
% end of file