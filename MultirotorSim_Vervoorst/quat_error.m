% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Compute quaternion error
% Generate the quaternion error between desired attitude quaternion
% and actual attitude quaternion.
%--------------------------------------------------------------------------
function quat_output = quat_error(quat_input)
%-------------------------------------------------------
% System states
q0  = quat_input(1);  % Actual attitude quaternion
q1  = quat_input(2);  % Actual attitude quaternion
q2  = quat_input(3);  % Actual attitude quaternion
q3  = quat_input(4);  % Actual attitude quaternion
qd0 = quat_input(5);  % Desired attitude quaternion
qd1 = quat_input(6);  % Desired attitude quaternion
qd2 = quat_input(7);  % Desired attitude quaternion
qd3 = quat_input(8);  % Desired attitude quaternion
%-------------------------------------------------------
% The definition of quaternion error used here is equivalent with:
% error_qdconj_q = quatmultiply(quatconj(qd'),q')';
qe = [qd0 qd1 qd2 qd3;-qd1 qd0 qd3 -qd2;-qd2 -qd3 qd0 qd1;-qd3 qd2 -qd1 qd0]*[q0;q1;q2;q3];
% The above expression multiplied out:
% qe(1) = q0*qd0 + q1*qd1 + q2*qd2 + q3*qd3;
% qe(2) = q1*qd0 - q0*qd1 + q2*qd3 - q3*qd2;
% q3(3) = q2*qd0 - q0*qd2 - q1*qd3 + q3*qd1;
% qe(4) = q1*qd2 - q0*qd3 - q2*qd1 + q3*qd0;
%-------------------------------------------------------
if (qe(1) < 0.0)       % q0 always positive for uniqueness of the quaternion
    qe(1) = -qe(1);
	qe(2) = -qe(2);
	qe(3) = -qe(3);
	qe(4) = -qe(4);
end
%-------------------------------------------------------
% Outputs are the four entries of the quaternion error
quat_output(1) = qe(1);
quat_output(2) = qe(2);
quat_output(3) = qe(3);
quat_output(4) = qe(4);
%--------------------------------------------------------------------------
% end of file