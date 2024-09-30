% Your data
for i= 1: length(pole_length)
plot(pole_length{i}, reward{i}, '*');
end
xlabel('Pole Length');
ylabel('Reward');
ylim([3000 5500]);
xlim([0 3]);
