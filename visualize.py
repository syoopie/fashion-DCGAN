import matplotlib.pyplot as plt
import pandas as pd


train_data = pd.read_csv('./data/train.csv')
X_train = train_data.drop('label', axis=1)
X_train = X_train.values
X_train = X_train.reshape(-1, 28, 28, 1)

fig, axes = plt.subplots(5, 5)
idx = 0
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(X_train[idx].reshape(28, 28), cmap='gray')
        idx += 1
        
for i, ax in enumerate(axes.flat):
    ax.set_axis_off()

plt.show()
