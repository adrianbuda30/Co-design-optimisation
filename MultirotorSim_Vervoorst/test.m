clear all;

% Provided data
Motor_arm_angle = [45 135 225 315] * pi/180; 
arm_length = [0.17 0.17 0.17 0.17]; 
arm_radius = [0.0013 0.0013 0.0013 0.0013]; 
prop_diameter = [0.2032 0.2032 0.2032 0.2032]; 
prop_height = [-0.028 -0.028 -0.028 -0.028]; 


Rz = zeros(3, 3, 4);
Motor_arm_angle_rad = [45 135 225 315] * pi/180; 

for i = 1:4
   Rz(:,:,i) = [cos(Motor_arm_angle_rad(i)) -sin(Motor_arm_angle_rad(i)) 0; sin(Motor_arm_angle_rad(i)) cos(Motor_arm_angle_rad(i)) 0; 0 0 1];
end

propeller_mass = prop_diameter .* (0.0110/ 0.2032);
total_propeller_mass = sum(propeller_mass);

volume = pi .* arm_length .* arm_radius.^2;
airframe_mass = 1700 * volume;
total_airframe_mass = sum(airframe_mass);

mass_center = 0.01;
COM_mass_center = [0 0 0.0014];

total_mass = mass_center + total_airframe_mass + total_propeller_mass

COM_propeller = zeros(4, 3);
COM_airframe = zeros(4, 3);

for i = 1:4
    COM_propeller(i,:) = Rz(:,:,i) * [arm_length(i) 0 prop_height(i)]';
    COM_airframe(i,:) = Rz(:,:,i) * [arm_length(i)/2 0 0]';
end

% Computing the weighted sum for center of mass
COM_propeller_total = [sum(propeller_mass' .* COM_propeller(:,1)); sum(propeller_mass' .* COM_propeller(:,2)); sum(propeller_mass' .* COM_propeller(:,3))];
COM_airframe_total = [sum(airframe_mass' .* COM_airframe(:,1)); sum(airframe_mass' .* COM_airframe(:,2)); sum(airframe_mass' .* COM_airframe(:,3))];
COM_center_total = mass_center .* COM_mass_center';  

COM_system_inter = (COM_propeller_total + COM_airframe_total + COM_center_total) / (total_propeller_mass + total_airframe_mass + mass_center);
COM_system = COM_system_inter(:,1)
% Inertia calculation
I_total = zeros(3, 3);

% ... [previous code]

for i = 1:4
    % Propeller
    I_propeller_cm = diag([1/12*propeller_mass(i)*arm_length(i)^2, 1/12*propeller_mass(i)*arm_length(i)^2, 0]);
    d_propeller = COM_propeller(i,:)' - COM_system; 
    
    d_propeller_outer_product = d_propeller * d_propeller'; 
    scalar_part = propeller_mass(i) * dot(d_propeller, d_propeller);
    I_propeller_total = I_propeller_cm + scalar_part .* eye(3) - propeller_mass(i) * d_propeller_outer_product;
    
    % Airframe
    I_cylinder_longitudinal = 1/2 * airframe_mass(i) * arm_radius(i)^2;
    I_cylinder_perpendicular = 1/4 * airframe_mass(i) * arm_radius(i)^2 + 1/12 * airframe_mass(i) * arm_length(i)^2;
    I_airframe_cm = diag([I_cylinder_perpendicular, I_cylinder_perpendicular, I_cylinder_longitudinal]);
    d_airframe = COM_airframe(i,:)' - COM_system;
    
    d_airframe_outer_product = d_airframe * d_airframe'; 
    scalar_part_airframe = airframe_mass(i) * dot(d_airframe, d_airframe);
    I_airframe_total = I_airframe_cm + scalar_part_airframe .* eye(3) - airframe_mass(i) * d_airframe_outer_product;
    
end
inertial_matrix_inter = diag(I_total + I_propeller_total + I_airframe_total);
% inertial_matrix = I_total + I_propeller_total + I_airframe_total;
% disp(inertial_matrix)

inertial_matrix = [inertial_matrix_inter(1) 0 0;
                    0 inertial_matrix_inter(2) 0;
                    0 0 inertial_matrix_inter(3)]

Surface_params = [arm_length(1) + arm_length(3); arm_length(2) + arm_length(4); 0.05]      % Combine the surface area parameters ellipsoid
% ... [rest of the code]
