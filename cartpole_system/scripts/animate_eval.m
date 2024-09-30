% clear all
% load("EVAL_CartPoleEnv_Tanh_Tsteps_122880000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_polemass_GaussMix_10.mat")
% Pole length and initial positions
l = 8; 
cart_position = out.cart_position;
pole_position = out.pole_position;
% Initialize the plot
figure;
cart = line([cart_position(1), cart_position(1)], [0, 0], 'Color', 'b', 'Marker', 's', 'LineWidth', 5, 'MarkerSize', 20);
pole = line([cart_position(1), pole_position(1,1)], [0, pole_position(1,2)], 'Color', 'r', 'LineWidth', 2);
axis([-2, 2, -2, 2]);
axis equal;

% Comment out or remove the next line to disable the grid
% grid on;

% Ensure white background (this is optional as the default is already white)
set(gca, 'Color', 'white');

% Add a placeholder for reward and length text
% rewardText = text(-1.8, 1.8, sprintf('Reward: %.2f\neffort: %.2f\nPole length: %.2f', rewards(1), effort(1), pole_length(1)), 'FontSize', 15);

% Animate the cart-pole
for k = 2:1:length(cart_position)
    cart.XData = [cart_position(k), cart_position(k)];
    pole.XData = [cart_position(k), pole_position(k,1)];
    pole.YData = [0, pole_position(k,2)];

    % Update the reward and length text
    % rewardText.String = sprintf('Reward: %.2f\neffort: %.2f\nPole length: %.2f', rewards(k), effort(k), pole_length(k));
    
    % Make sure text position remains constant relative to the moving axes
    % rewardText.Position = [cart_position(k) - 2, 2, 0];

    drawnow;
end
plot(steps,pole_length, '*')