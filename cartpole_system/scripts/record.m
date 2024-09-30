clear all;
load("EVAL_CartPoleEnv_Tanh_Tsteps_122880000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_polemass_GaussMix_10.mat")

% Pole length and initial positions
l = 8;

% Initialize the plot
figure;
cart = line([cart_position(1), cart_position(1)], [0, 0], 'Color', 'b', 'Marker', 's', 'LineWidth', 5, 'MarkerSize', 20);
pole = line([cart_position(1), pole_position(1,1)], [0, pole_position(1,2)], 'Color', 'r', 'LineWidth', 2);
axis([-2, 2, -2, 2]);
axis equal;
set(gca, 'Color', 'white');

% Add a placeholder for reward and length text
rewardText = text(-1.8, 1.8, sprintf('Reward: %.2f\neffort: %.2f\nPole length: %.2f', rewards(1), effort(1), pole_length(1)), 'FontSize', 10);

% Set up video recording
v = VideoWriter('cart_pole_animation.avi', 'Uncompressed AVI');
open(v);

% Animate the cart-pole
for k = 2:10:length(cart_position)
    cart.XData = [cart_position(k), cart_position(k)];
    pole.XData = [cart_position(k), pole_position(k,1)];
    pole.YData = [0, pole_position(k,2)];
    
    % Update the reward and length text
    rewardText.String = sprintf('Reward: %.2f\neffort: %.2f\nPole length: %.2f', rewards(k), effort(k), pole_length(k));
    rewardText.Position = [cart_position(k) - 1.8, 1.8, 0];

    drawnow;
    
    % Capture the current frame and write to video
    frame = getframe(gcf);
    writeVideo(v, frame);
end

plot(steps, pole_length, '*');

% Close the video file
close(v);
