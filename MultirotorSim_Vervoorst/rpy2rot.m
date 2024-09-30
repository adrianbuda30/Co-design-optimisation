% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Generate the rotation matrix for a given set of Euler angles
%  This files takes Roll/Pitch/Yaw Euler angles as inputs
% and transforms into the rotation matrix representation.
% Representative of Euler angle sequence (1,2,3) according to
% "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
% by James Diebel, Stanford University, 20 October 2006
%--------------------------------------------------------------------------
function R_123 = rpy2quat(rot_angles)
%-------------------------------------------------------
% System states
phi   = rot_angles(1);  % Roll input [rad]
theta = rot_angles(2);  % Pitch input [rad]
psi   = rot_angles(3);  % Yaw input [rad]
%-------------------------------------------------------
% Outputs are the 3x3 rotation matrix R_123
% This represents the rotation matrix R_IB, from Inertial axes to Body axes
R_123 = [cos(theta)*cos(psi) cos(theta)*sin(psi) -sin(theta); ...
    sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) cos(theta)*sin(phi); ...
    cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi) cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi) cos(theta)*cos(phi)];
%-------------------------------------------------------

%--------------------------------------------------------------------------
% end of file