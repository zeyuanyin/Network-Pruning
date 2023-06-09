# The code is generated by ChatGPT

import matplotlib.pyplot as plt

# Data for ResNet18
resnet18_pruning_ratio = [0, 10, 20, 30, 40, 50]
resnet18_accuracy = [69.76, 67.99, 59.36, 49.99, 35.35, 17.35]

# Data for ResNet50
resnet50_pruning_ratio = [0, 10, 20, 30, 40, 50, 60, 70]
resnet50_accuracy = [76.15, 76.14, 76.06, 75.66, 75.05, 73.26, 64.75, 25.43]

# Plotting
plt.plot(resnet18_pruning_ratio, resnet18_accuracy, marker='o', linestyle='-', label='ResNet18')
plt.plot(resnet50_pruning_ratio, resnet50_accuracy, marker='o', linestyle='-', label='ResNet50')
plt.xlabel('Pruning Ratio (%)')
plt.ylabel('Acc@1 (%)')
plt.title('Pruning Ratio vs. Top1 Accuracy')
plt.legend()
plt.grid(True)

# Add text annotations for top-1 accuracy values
for x, y in zip(resnet18_pruning_ratio, resnet18_accuracy):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

for x, y in zip(resnet50_pruning_ratio, resnet50_accuracy):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

# Save the plot as an image file
plt.savefig('plot.png', bbox_inches='tight')
