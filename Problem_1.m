
% EECE 5644 Introduction to Machine Learning and Pattern Recognition
% Northeastern University, Summer I 2023
% Ergest Beshaj 

clear
clc
close all

%% Problem 1

% Define parameters for each class
mu0 = [-1; 1; -1; 1]; % Mean of class 0
Sigma0 = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2]; % Covariance matrix of class 0
mu1 = [1; 1; 1; 1]; % Mean of class 1
Sigma1 = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3]; % Covariance matrix of class 1

% Define prior probabilities of each class
PY0 = 0.7;
PY1 = 0.3; 

% Generate samples for the random vector X
N = 10000; % Number of samples to generate

% array to hold samples and class labels
X = zeros(4,N);
labels = zeros(1,N);

% generate samples 
for i = 1:N
    if rand < PY0
        % class 0
        X(:, i) = mvnrnd(mu0, Sigma0);
        labels(i) = 0;
    else
        % class 1
        X(:, i) = mvnrnd(mu1, Sigma1);
        labels(i) = 1;
    end
end


%% PART 1-A

% Compute the likelihood ratio
pdf0 = mvnpdf(X', mu0', Sigma0);
pdf1 = mvnpdf(X', mu1', Sigma1);
likelihood_ratio =  pdf1 ./ pdf0;

% Loss matrix
% Assume 0-1 loss scenario 
cost01 = 1;
cost10 = 1;

% The entry at row i, column j is the cost of classifying a sample from class i-1 as class j-1
L = [0, cost01;  % cost01 is the cost of misclassifying a class 0 sample as class 1
    cost10, 0]; % cost10 is the cost of misclassifying a class 1 sample as class 0

% Determine the theoretical threshold gamma
gamma = (PY0 * (L(2,1) - L(1,1))) / (PY1 * (L(1, 2) - L(2,2)));
theoretical_gamma = gamma;

%% Plot the PDFs. Record the decisions. Calculate the risk for both cases. Find the min risk


% Plot the PDFs - this is just for us to verify the dataset

% Create a range of values for X to calculate the PDFs
X_range = linspace(-5, 5, 100);

% Create a 2-by-2 subplot layout
figure;
for i = 1:4
    subplot(2, 2, i);
    
    % Calculate the PDFs for the ith dimension
    pdf0_dim = normpdf(X_range, mu0(i), sqrt(Sigma0(i, i)));
    pdf1_dim = normpdf(X_range, mu1(i), sqrt(Sigma1(i, i)));

    % Plot the PDFs for the ith dimension
    plot(X_range, pdf0_dim, 'r');
    hold on;
    plot(X_range, pdf1_dim, 'b');
    
    % Set the title and labels
    title(['Dimension ' num2str(i)]);
    xlabel('X');
    ylabel('Probability Density');
    
    % Set the legend
    legend('Class 0', 'Class 1');
end

% Initialize an array to hold the class predictions with the min risk
min_risk_datapoints = zeros(1,N);
risk_classify_as_0 = zeros(1,N);
risk_classify_as_1 = zeros(1,N);
% Loop through each sample
for i = 1:N
    % Calculate the risk for classifying the sample as class 0
    risk_classify_as_0(i) = L(1,1) * PY0 * pdf0(i) + L(1,2) * PY1 * pdf1(i);
    
    % Calculate the risk for classifying the sample as class 1
    risk_classify_as_1(i) = L(2,1) * PY0 * pdf0(i) + L(2,2) * PY1 * pdf1(i);
    
    % Choose the class with the minimum risk
    if risk_classify_as_0(i) < risk_classify_as_1(i)
        min_risk_datapoints(i) = 0;
    else
        min_risk_datapoints(i) = 1;
    end
end


% Classify samples based on likelihood ratio and threshold
decisions = likelihood_ratio > gamma;
decisions = double(decisions); % Convert to numeric value 

T = table(decisions, risk_classify_as_0', risk_classify_as_1', min_risk_datapoints', ...
    'VariableNames', {'Decision', 'Risk_Classify_as_0', 'Risk_Classify_as_1', 'Class_Min_Risk'});

% Display the table - uncomment to show the table
 % disp(T);



%% PART 2-A, 3-A

% calculate the true positive rate, false positive rate, 
% false negative rate, and true negative rate. 

% Initialize the true positive and false positive rates

n = 100; % infinity
d = 0.01; % incremental step

true_positive_v = zeros(1, n/d);
false_positive_v = zeros(1, n/d);

gammas = zeros(1, n/d);
i=1; % index to store the TPR and FPR in vector

% set a min error very high in order to  be updated later 
min_error = Inf;

% set gamma optimal at 0 to start 
gamma_optimal = 0;

% Vary the threshold gamma
for gamma = 0:d:n
% gammas is used to store the range of gamma to be plotted in 3D ROC
gammas(i) = gamma;

% update the decision in response to changes in gamma 
decisions = likelihood_ratio > gamma;

    % Calculate

    % count the number of samples where the actual label is 1 (positive) 
    % and the predicted label is also 1 (positive).
    true_positive = sum(decisions(labels == 1)== 1);

    % count the number of samples where the actual label is 0 (negative) 
    % but the predicted label is 1 (positive).
    false_positive = sum(decisions(labels == 0)== 1);

    % count the number of samples where the actual label is 0 (negative) 
    % and the predicted label is also 0 (negative). 
    true_negative = sum(~decisions(labels == 0));

    % count the number of samples where the actual label is 1 (positive) 
    % but the predicted label is 0 (negative).
    false_negative = sum(~decisions(labels == 1));
    
    true_positive_rate = true_positive / (true_positive + false_negative);
    false_positive_rate = false_positive / (false_positive + true_negative);
    false_negative_rate = false_negative / (false_negative + true_positive);

    % Append the TPR and FPR to their respective vectors
    true_positive_v(i) = true_positive_rate;
    false_positive_v(i) = false_positive_rate;

    error_probability = false_positive_rate * PY0 + false_negative_rate * PY1;

    if error_probability < min_error
        min_error = error_probability;
        gamma_optimal = gamma;
    end

    i=i+1;
end

% Plot the ROC curve
figure;
plot(false_positive_v, true_positive_v);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;


% calculate the true positive and false positive rate for the optimal gamma
decisions_opt = likelihood_ratio > gamma_optimal;
true_positive_opt = sum(decisions_opt(labels == 1)== 1);
false_positive_opt = sum(decisions_opt(labels == 0)== 1);
true_negative_opt = sum(~decisions_opt(labels == 0));
false_negative_opt = sum(~decisions_opt(labels == 1));

true_positive_rate_opt = true_positive_opt / (true_positive_opt + false_negative_opt);
false_positive_rate_opt = false_positive_opt / (false_positive_opt + true_negative_opt);

% plot the optimal gamma on the ROC curve
hold on;
plot(false_positive_rate_opt, true_positive_rate_opt, 'go', 'MarkerSize', 8);
hold off;


% Plot the 3D ROC curve
figure;
plot3(false_positive_v, true_positive_v,gammas);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
zlabel('Threshold (\gamma)');
title('3D ROC Curve');

% the optimal gamma on the 3D ROC curve
hold on;
plot3(false_positive_rate_opt, true_positive_rate_opt, gamma_optimal, 'go', 'MarkerSize', 8);
hold off;

fprintf('At gamma = 0, the ROC curve is at :  %.0f , %.0f.\n', false_positive_v(1), true_positive_v(1));
fprintf('At gamma = infinit, the ROC curve is at :  %.0f , %.0f.\n', false_positive_v(n/d), true_positive_v(n/d));
fprintf('Optimal gamma that minimizes the error: %.3f\n', gamma_optimal);
fprintf('Theoretical gamma is: %.3f\n', theoretical_gamma);
fprintf('P(error)_min = %.3f\n', min_error);


%% Summary 

% The obtained threshold value (gamma) is not precisely the one calculated 
% theoretically, given the defined parameters such as mean, covariance, 
% and priors. This discrepancy is due to the randomness introduced while 
% drawing samples from the distributions defined by these parameters. 
% Consequently, every time the script runs, it generates varying samples. 
% 
% This variance influences other components as well such as likelihood 
% ratios, decisions, true and false positive rates, among others.
% Therefore, the resulting gamma value, although not identical, hover 
% around the theoretical value.


%% PART B
% Define the covariance matrix of the Naive Bayes Classifier
Sigma0_NB = diag(diag(Sigma0));
Sigma1_NB = diag(diag(Sigma1));

% Compute the likelihood ratio for the Naive Bayes Classifier
pdf0_NB = mvnpdf(X', mu0', Sigma0_NB);
pdf1_NB = mvnpdf(X', mu1', Sigma1_NB);
likelihood_ratio_NB =  pdf1_NB ./ pdf0_NB;

true_positive_v_NB = zeros(1, n/d);
false_positive_v_NB = zeros(1, n/d);
gammas_NB = zeros(1, n/d);
j=1;
min_error_NB = Inf;
gamma_optimal_NB = 0;

for gamma_NB = 0:d:n
    gammas_NB(j) = gamma_NB;
    decisions_NB = likelihood_ratio_NB > gamma_NB;

    true_positive_NB = sum(decisions_NB(labels == 1)== 1);
    false_positive_NB = sum(decisions_NB(labels == 0)== 1);
    true_negative_NB = sum(~decisions_NB(labels == 0));
    false_negative_NB = sum(~decisions_NB(labels == 1));
    
    true_positive_rate_NB = true_positive_NB / (true_positive_NB + false_negative_NB);
    false_positive_rate_NB = false_positive_NB / (false_positive_NB + true_negative_NB);
    false_negative_rate_NB = false_negative_NB / (false_negative_NB + true_positive_NB);

    true_positive_v_NB(j) = true_positive_rate_NB;
    false_positive_v_NB(j) = false_positive_rate_NB;

    error_probability_NB = false_positive_rate_NB * PY0 + false_negative_rate_NB * PY1;

    if error_probability_NB < min_error_NB
        min_error_NB = error_probability_NB;
        gamma_optimal_NB = gamma_NB;
    end

    j=j+1;
end

% Plot the ROC curve for the Naive Bayes Classifier
figure;
plot(false_positive_v_NB, true_positive_v_NB);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Naive Bayes Classifier');
grid on;

% calculate the true positive and false positive rate for the optimal gamma for Naive Bayes
decisions_opt_NB = likelihood_ratio_NB > gamma_optimal_NB;
true_positive_opt_NB = sum(decisions_opt_NB(labels == 1)== 1);
false_positive_opt_NB = sum(decisions_opt_NB(labels == 0)== 1);
true_negative_opt_NB = sum(~decisions_opt_NB(labels == 0));
false_negative_opt_NB = sum(~decisions_opt_NB(labels == 1));

true_positive_rate_opt_NB = true_positive_opt_NB / (true_positive_opt_NB + false_negative_opt_NB);
false_positive_rate_opt_NB = false_positive_opt_NB / (false_positive_opt_NB + true_negative_opt_NB);

% plot the optimal gamma on the ROC curve
hold on;
plot(false_positive_rate_opt_NB, true_positive_rate_opt_NB, 'go', 'MarkerSize', 10);
hold off;

% Plot the 3D ROC curve
figure;
plot3(false_positive_v, true_positive_v, gammas);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
zlabel('Threshold (\gamma)');
title('3D ROC Curve');

% the optimal gamma on the 3D ROC curve
hold on;
plot3(false_positive_rate_opt_NB, true_positive_rate_opt_NB, gamma_optimal_NB, 'go', 'MarkerSize', 10);
hold off;

fprintf('\nNaive Bayes Classifier\n')

fprintf('At gamma = 0, the ROC curve is at :  %.0f , %.0f.\n', false_positive_v(1), true_positive_v(1));
fprintf('At gamma = infinit, the ROC curve is at :  %.0f , %.0f.\n', false_positive_v(n/d), true_positive_v(n/d));
fprintf('Optimal gamma that minimizes the error: %.3f\n', gamma_optimal_NB);
fprintf('P(error)_min = %.3f\n', min_error);

% Plotting the first ROC
plot(false_positive_v, true_positive_v, 'b');
hold on;

% Plotting the second ROC
plot(false_positive_v_NB, true_positive_v_NB, 'r'); 

% Adding labels and title
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve Comparison');
grid on;

% Adding a legend
legend('ROC_A', 'ROC_B');


