% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Compute the quaternion product
% Compute the product between quaternions q1 and q2
%--------------------------------------------------------------------------
function q = quat_mult(q1,q2)
q = [q1(1)*q2(1)-q1(2:4)'*q2(2:4);q1(1)*q2(2:4)+q2(1)*q1(2:4)+cross(q1(2:4),q2(2:4))];
%--------------------------------------------------------------------------
% end of file