import numpy as np
import matplotlib.pyplot as plt

base_lr = 1e-3
num_epochs = 30
t_max = 5
eta_min = 0

epochs = []
lr_values = []


for i in range(num_epochs + 1):
    lr_values += [eta_min + 0.5 * (base_lr - eta_min) * (1 + np.cos(np.pi * i / 5))]
    epochs += [i]
    
plt.plot(epochs, lr_values)

plt.xlabel('epoch')
plt.ylabel('learning_rate')
plt.title('Cosine Annealing')

# Show the plot
plt.savefig("testing/lr_graph.png")
    
