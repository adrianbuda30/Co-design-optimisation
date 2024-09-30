% Jan Vervoorst, vervoors@illinois.edu
% University of Illinois 2015/16
%--------------------------------------------------------------------------
%% Compute desired motor rpm from thrust command
% Using a nominal second order model of the thrust equation, compute the
% desired motor RPM from a commanded thrust input.
%--------------------------------------------------------------------------
function rpm_output = thrust2rpm(thrust_input)
%-------------------------------------------------------
% System states
a1     = thrust_input(1);  % Linear thrust parameter a1
a2     = thrust_input(2);  % Quadratic thrust parameter a2
thrust = thrust_input(3);  % Desired thrust command
%-------------------------------------------------------
% Motor RPM and thrust are related by a quadratic function of the form
% thrust = a1 * RPM + a2 * RPM^2
% This quadratic equation is then solved for the RPM value
% (and only the positive solution is considered):
rpm_output = -a1/(2*a2) + sqrt((a1/(2*a2))^2 + thrust/a2);
%--------------------------------------------------------------------------
% end of file