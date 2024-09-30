% Assuming you have 'pole_length' as input and 'average_effort' as output
inputs = pole_length.';
targets = average_effort.';

% Define the network architecture
hiddenLayerSize = 10;  % You can adjust this
net = fitnet(hiddenLayerSize);

% Divide dataset into training, validation, and testing
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
[net, tr] = train(net, inputs, targets);

% Predict using the trained network
predictions = net(inputs);

% View the network
view(net)

% Plot the results
figure;
plot(inputs, targets, '*', inputs, predictions, '-');
legend('Data', 'Neural Network Fit');
