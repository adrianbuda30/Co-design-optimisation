
cart_position = out.cart_position;
pole_position = out.pole_position;
% Pole length and initial positions
l = 8; 
% Initialize the plot
figure;
% Create initial cart as a rectangle [x, y, width, height]
cart = rectangle('Position', [cart_position(1)-0.25, -0.15, 0.05, 0.03], 'EdgeColor', 'b', 'FaceColor', 'b');
% Create initial pole as a line
pole = line([cart_position(1), pole_position(1,1)], [0, pole_position(1,2)], 'Color', 'r', 'LineWidth', 6);

axis([-1, 1, -1, 1]);
axis equal;

% Ensure white background (this is optional as the default is already white)
set(gca, 'Color', 'white');

% Animate the cart-pole
for k = 2:1:length(cart_position)
    % Update cart position
    cart.Position = [cart_position(k)-0.1, -0.1, 0.2, 0.2];

    % Update pole position
    pole.XData = [cart_position(k), pole_position(k,1)];
    pole.YData = [0, pole_position(k,2)];

    drawnow;
end
