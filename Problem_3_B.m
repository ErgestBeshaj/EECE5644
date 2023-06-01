
% EECE 5644 Introduction to Machine Learning and Pattern Recognition
% Northeastern University, Summer I 2023
% Ergest Beshaj 

clear
clc
close all

%% Problem 3-B

% Load the dataset
training_dataset = readtable('X_train.txt');
training_dataset_labels = readtable('y_train.txt');
testing_dataset = readtable('X_test.txt');
testing_dataset_labels = readtable('y_test.txt');

% Extract features and labels 
training_features = table2array(training_dataset);
training_labels = table2array(training_dataset_labels);

testing_features = table2array(testing_dataset);
testing_labels = table2array(testing_dataset_labels);

% Find the unique class labels (in ascending order)
class_samples_unique = unique(training_labels);

% Calculate sample means and covariance matrices for each class.
% The rows correspond to a class and the columns to features.
means = zeros(length(class_samples_unique), size(training_features, 2));

% covariances is a 3D array to store the covariance matrices for each class. .
covariances = zeros(size(training_features, 2), size(training_features, 2), length(class_samples_unique));

lambda = 0.02;  %  adjusting this value for regularization

% calculate the mean and covariance matrix for each class
for i = 1:length(class_samples_unique)
    % Get the current class label.
    current_class = class_samples_unique(i);

    % find only the training samples belonging to the current class i
    class_samples = training_features(training_labels == current_class, :);

    % Compute the mean and covariance of the current class samples
    means(i, :) = mean(class_samples, 1);

    covariances(:, :, i) = cov(class_samples);
    
    % Regularize the covariance matrix
    covariances(:, :, i) = covariances(:, :, i) + lambda * eye(size(covariances(:, :, i)));   
end

% Calculate class priors
class_priors = histcounts(training_labels, 'Normalization', 'probability');

% Classification
predicted_labels = zeros(size(testing_labels));

for i = 1:size(testing_features, 1)  % Loop over each sample in the test set
    max_posterior = -100000;  % set the max posterior very low
    for j_feature = 1:length(class_samples_unique)  % Loop over each class
        % Compute the conditional pdf
        posterior = mvnpdf(testing_features(i, :), means(j_feature, :), covariances(:, :, j_feature)) * class_priors(j_feature);
        if posterior > max_posterior 
            max_posterior = posterior;  % Update 
            predicted_labels(i) = class_samples_unique(j_feature);  % Update the predicted label
        end
    end
end

% Compute the error rate
error_rate = sum(predicted_labels ~= testing_labels) / size(testing_labels, 1);

% Display
disp(['Error rate: ', num2str(error_rate)]);

% The confusion matrix
confusion_matrix = confusionmat(testing_labels, predicted_labels);

% Display the confusion matrix
disp('Confusion matrix:');
disp(confusion_matrix);

% Compute PCA
[coeff,score,~] = pca(training_features);

figure
for class_index = 1:length(class_samples_unique)
    % Create a new subplot for this class
    subplot(4, 2, class_index)

    % Get indices for the current class
    class_indices = (training_labels == class_samples_unique(class_index));

    % Scatter plot for current class
    scatter(score(class_indices,1), score(class_indices,2), 8, 'filled')

    title(['Projection onto the first two principal components - Class ', num2str(class_samples_unique(class_index))])
    xlabel('1st Principal Component')
    ylabel('2nd Principal Component')
end

j_feature = 4;
     figure
    for class_index = 1:3
        subplot(3, 1, class_index)
        % The current class
        current_class_samples = training_features(training_labels == class_samples_unique(class_index), :);
        % Select the feature
        feature = current_class_samples(:, j_feature);
        % Plot a histogram
        histogram(feature, 'Normalization', 'pdf');
        hold on
        % Plot a Gaussian PDF
        mu = means(class_index, j_feature);
        sigma = sqrt(covariances(j_feature, j_feature, class_index));
        x = linspace(mu - 5*sigma, mu + 5*sigma, 100);  
        y = normpdf(x, mu, sigma);  % Create the Gaussian PDF
        plot(x, y, 'LineWidth', 3);
        hold off
        title(['Class ', num2str(class_samples_unique(class_index)), ' - Feature ' num2str(j_feature)])
    end

