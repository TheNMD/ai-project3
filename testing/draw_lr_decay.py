import numpy as np
import matplotlib.pyplot as plt

groups = ["head"]
lr_values = [0.001, 0.0008, 0.00064, 0.0005120000000000001, 0.0004096000000000001, 0.0003276800000000001, 0.0002621440000000001, 0.00020971520000000012, 0.0001677721600000001, 0.00013421772800000008, 0.00010737418240000007, 8.589934592000007e-05, 6.871947673600006e-05, 5.497558138880005e-05, 4.3980465111040044e-05, 3.5184372088832036e-05, 2.814749767106563e-05]


for i in range(1, len(lr_values) - 1):
    groups += [f"g{i}"]
else:
    groups += ["stem"]

plt.plot(groups, lr_values)

plt.xlabel('groups')
plt.ylabel('learning_rate')
plt.title('Layer-wise learning rate decay')

# Show the plot
plt.savefig("testing/lr_decay_graph.png")
    
