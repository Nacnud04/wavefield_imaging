% define function parameters
alpha = 0.90; % dampening parameter
nb = 64; % size of buffer

% Generate a range of values for i
i_values = linspace(-nb, nb, 100); % i is the depth into the buffer region.

% Calculate the corresponding values for the first C function
damp2_values = exp(-1.0 * abs((power((i_values - 1.0), 2) * log(alpha)) / (power(nb, 2))));

% Calculate the corresponding values for the second C function
damp1_values = exp(-1.0 * abs(((i_values - 1.0) * log(alpha)) / nb));

% Plot both functions on the same graph
plot(i_values, damp2_values, 'LineWidth', 2, 'DisplayName', 'Function 2');
hold on;
plot(i_values, damp1_values, 'LineWidth', 2, 'DisplayName', 'Function 1');
xlabel('i');
ylabel('damp');
title('Sponges');
grid on;
legend('show');
hold off;

%% 
