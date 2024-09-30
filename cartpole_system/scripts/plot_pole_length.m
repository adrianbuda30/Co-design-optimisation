value_arr = [50, 100, 500, 1000, 2500];
figure;
for j = 1:5
value = value_arr(j);
for i = 1:8
    length_all((i-1)*value+1:i*value) = pole_length{i}(1:value);
end
subplot(1,5,j)
mean(length_all)
std(length_all)
histogram(length_all, 'BinWidth', 1); % Change BinWidth according to your needs
xlabel('Value');
ylabel('Frequency');
titleText = sprintf('First %d values', value*8);
title(titleText);
clear value length_all;
end
clear all