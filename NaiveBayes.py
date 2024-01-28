import numpy as np
import matplotlib.pyplot as plt

def naive_bayes(train_data, train_labels, test_data):
    num_samples, num_features = train_data.shape
    num_classes = len(np.unique(train_labels))
    
    class_probs = np.zeros(num_classes)
    feature_probs = np.zeros((num_classes, num_features))
    
    for c in range(num_classes):
        class_mask = (train_labels == c)
        class_probs[c] = np.sum(class_mask) / num_samples
        feature_probs[c, :] = np.mean(train_data[class_mask, :], axis=0)
    
    predictions = np.argmax(np.log(class_probs) + np.dot(test_data, np.log(feature_probs).T), axis=1)
    
    return predictions


np.random.seed(42)
mean1 = [1, 1]
cov1 = [[1, 0], [0, 1]]
class1_data = np.random.multivariate_normal(mean1, cov1, 50)
mean2 = [5, 5]
cov2 = [[1, 0], [0, 1]]
class2_data = np.random.multivariate_normal(mean2, cov2, 50)
data = np.vstack((class1_data, class2_data))
labels = np.concatenate((np.zeros(50), np.ones(50)))
split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]
train_labels, test_labels = labels[:split], labels[split:]
predictions = naive_bayes(train_data, train_labels, test_data)
plt.figure(figsize=(8, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='viridis', marker='o', edgecolors='k', label='Training Points')
plt.scatter(test_data[:, 0], test_data[:, 1], c=predictions, cmap='viridis', marker='x', s=200, linewidths=2, label='Predictions')
plt.title('Naive Bayes Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
