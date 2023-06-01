% EECE 5644 Introduction to Machine Learning and Pattern Recognition
% Northeastern University, Summer I 2023
% Ergest Beshaj 

clear
clc
close all

%% Problem 2

% mu1, mu2, mu3, and mu4 represent the means of the four Gaussian distributions,
% and Sigma1, Sigma2, Sigma3, and Sigma4 represent the covariance matrices 
% for these distributions. 
% 
% The means were chosen as [1 1 1], [3 3 3], [5 5 5] and [7 7 7] to 
% ensure that they are well separated in the 3-dimensional space, making it 
% easier to distinguish between the classes. This separation also meets the
% problem's criteria that the distance between the means should be twice 
% the average standard deviation of the Gaussian components.
% 
% The covariance matrices are chosen to be the identity matrix eye(3), 
% which means that the variances of the variables are 1 and the covariance 
% between any two different variables is 0. This corresponds to an isotropic 
% Gaussian distribution, where the spread of the distribution is the same 
% in all directions in the 3-dimensional space.

% Define the parameters of the Gaussians

mu1 = [1 1 1]; Sigma1 = eye(3);
mu2 = [3 3 3]; Sigma2 = eye(3);
mu3 = [5 5 5]; Sigma3 = eye(3);
mu4 = [7 7 7]; Sigma4 = eye(3);

% For testing purposes
% mu1 = [1 1 1]; Sigma1 = eye(3);
% mu2 = [5 5 5]; Sigma2 = eye(3);
% mu3 = [9 9 9]; Sigma3 = eye(3);
% mu4 = [13 13 13]; Sigma4 = eye(3);

% mu1 = [1 1 1]; Sigma1 = eye(3);
% mu2 = [7 7 7]; Sigma2 = eye(3);
% mu3 = [13 13 13]; Sigma3 = eye(3);
% mu4 = [19 19 19]; Sigma4 = eye(3);


% Define class priors
prior = [0.3, 0.3, 0.4];

% Generate 10000 samples
N = 10000;
labels = rand(1, N); % generate a random number from 0-1 to generate the samples from different classes 
X = zeros(3, N); % used to store the samples
true_labels = zeros(1, N); % used to keep track of the true labels 

for i = 1:N
    if labels(i) <= prior(1)
        X(:, i) = mvnrnd(mu1, Sigma1);
        true_labels(i) = 1;

    elseif labels(i) <= sum(prior(1:2))
        X(:, i) = mvnrnd(mu2, Sigma2);
        true_labels(i) = 2;
    else
        if rand <= 0.5 % equal weights 
            X(:, i) = mvnrnd(mu3, Sigma3);
        else
            X(:, i) = mvnrnd(mu4, Sigma4);
        end
        true_labels(i) = 3;
    end
end

% Implement MAP classifier
predicted_labels = zeros(1, N);
for i = 1:N
    pdf1 = mvnpdf(X(:, i)', mu1, Sigma1);
    pdf2 = mvnpdf(X(:, i)', mu2, Sigma2);
    pdf3 = 0.5 * mvnpdf(X(:, i)', mu3, Sigma3) + 0.5 * mvnpdf(X(:, i)', mu4, Sigma4); % class 3 is defined as a mixture of two Gaussian distributions with equal weights
    
     % Compute the posterior probabilities by multiplying with the class priors
    posterior1 = pdf1 * prior(1);
    posterior2 = pdf2 * prior(2);
    posterior3 = pdf3 * prior(3);
    
    % Assign the class with the maximum posterior probability
    % Store just the index of the max pdf (~ gets rid off the max pdf)
    [~, predicted_labels(i)] = max([posterior1, posterior2, posterior3]);
end

% Compute confusion matrix
confusion_matrix = confusionmat(true_labels, predicted_labels)

% Define markers and colors
markers = {'o', '^', 's'}; % circle, triangle, square
colors = {'r', 'g'}; % red, green

figure;  hold on;  grid on;
for true_label = 1:3
    for correct = [true, false]
        % Select points with the current true label and correct/incorrect prediction
        test = (true_labels == true_label) & ((predicted_labels == true_labels) == correct);
        if any(test)
            % Choose the color based on whether the point was classified correctly
            color = colors{correct + 1};
            % Choose the marker based on the true label
            marker = markers{true_label};
            % Plot the points with the chosen marker and color
            plot3(X(1, test), X(2, test), X(3, test), [color marker], 'MarkerSize', 5);
        end
    end
end

xlabel('X1');
ylabel('X2');
zlabel('X3');
title('3D Scatter Plot of the Data');
legend('Class 1 - Correct', 'Class 1 - Incorrect', 'Class 2 - Correct', 'Class 2 - Incorrect', 'Class 3 - Correct', 'Class 3 - Incorrect');
view(45, 45); % Set the view angle

%% PART B
% Loss matrices
Lambda10 = [0 1 10; 1 0 10; 1 1 0];
Lambda100 = [0 1 100; 1 0 100; 1 1 0];

% ERM Classification for Lambda10
risk = zeros(3,1); % a vector to store the calculated risk for each class

% A Nx1 matrix of zeros. This vector will store the predicted labels for 
% each data point based on the ERM with Lambda10 classification.
predicted_labels_ERM10 = zeros(N,1);

for i = 1:N
    pdf1 = mvnpdf(X(:, i)', mu1, Sigma1);
    pdf2 = mvnpdf(X(:, i)', mu2, Sigma2);
    pdf3 = 0.5 * mvnpdf(X(:, i)', mu3, Sigma3) + 0.5 * mvnpdf(X(:, i)', mu4, Sigma4);
    posterior = [pdf1*prior(1)  pdf2*prior(2) pdf3*prior(3)];
    
    for j = 1:3
        risk(j) = sum(Lambda10(j,:) .* posterior);
    end

    [~, predicted_labels_ERM10(i)] = min(risk); % find the class with min risk
end

% Compute confusion matrix
confusion_matrix_ERM10 = confusionmat(true_labels, predicted_labels_ERM10);


% ERM Classification for Lambda100
predicted_labels_ERM100 = zeros(N,1);

for i = 1:N
    pdf1 = mvnpdf(X(:, i)', mu1, Sigma1);
    pdf2 = mvnpdf(X(:, i)', mu2, Sigma2);
    pdf3 = 0.5 * mvnpdf(X(:, i)', mu3, Sigma3) + 0.5 * mvnpdf(X(:, i)', mu4, Sigma4);
    posterior = [pdf1*prior(1)  pdf2*prior(2) pdf3*prior(3)];

    for j = 1:3
        risk(j) = sum(Lambda100(j,:) .* posterior);
    end
    [~, predicted_labels_ERM100(i)] = min(risk);
end

% Compute confusion matrix
confusion_matrix_ERM100 = confusionmat(true_labels, predicted_labels_ERM100);

% Compute expected risk
% Lambda * confusion_matrix_ERM10 is the risk for each class.
% diag() shows only the risk for correct predictions.
% Sum() - Computes total risk for correct predictions.
% ... / N - to obtain average risk per sample (expected risk).
expected_risk_ERM10 = sum(diag(Lambda10 * confusion_matrix_ERM10)) / N;
expected_risk_ERM100 = sum(diag(Lambda100 * confusion_matrix_ERM100)) / N;

% Display numerical results
disp("Expected Risk with Lambda10:");
disp(expected_risk_ERM10);
disp("Expected Risk with Lambda100:");
disp(expected_risk_ERM100);

figure;  hold on;  grid on; 
for true_label = 1:3
    for correct = [true, false]
        test10 = (true_labels == true_label) & ((predicted_labels_ERM10' == true_labels) == correct); 
        if any(test10)
            color = colors{correct + 1};
            marker = markers{true_label};
            plot3(X(1, test10), X(2, test10), X(3, test10), [color marker], 'MarkerSize', 5);
        end
    end
end
xlabel('X1'); 
ylabel('X2'); 
zlabel('X3');
title('3D Scatter Plot of the Data (Lambda10)');
legend('Class 1 - Correct', 'Class 1 - Incorrect', 'Class 2 - Correct', 'Class 2 - Incorrect', 'Class 3 - Correct', 'Class 3 - Incorrect');
view(45, 45);

figure;  hold on;  grid on; title('3D Scatter Plot of the Data (Lambda100)');
for true_label = 1:3
    for correct = [true, false]
        test100 = (true_labels == true_label) & ((predicted_labels_ERM100' == true_labels) == correct);
        if any(test100)
            color = colors{correct + 1};
            marker = markers{true_label};
            plot3(X(1, test100), X(2, test100), X(3, test100), [color marker], 'MarkerSize', 5);
        end
    end
end
xlabel('X1'); ylabel('X2'); zlabel('X3');
legend('Class 1 - Correct', 'Class 1 - Incorrect', 'Class 2 - Correct', 'Class 2 - Incorrect', 'Class 3 - Correct', 'Class 3 - Incorrect');
view(45, 45); % Set the view angle



