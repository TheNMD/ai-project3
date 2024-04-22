import pandas as pd
import matplotlib.pyplot as plt

def plot_results(monitor_value="val_acc"):
  log_results = pd.read_csv("metrics.csv")
  train_results = log_results[['epoch', 'train_loss', 'train_acc']].dropna()
  train_results = train_results.groupby(['epoch'], as_index=False).mean()
  val_results = log_results[['epoch', 'val_loss', 'val_acc']].dropna()
  val_results = val_results.groupby(['epoch'], as_index=False).mean()

  if monitor_value == 'val_loss':
    min_idx = val_results['val_loss'].idxmin()
    best_epoch = val_results.loc[min_idx, 'epoch']
  elif monitor_value == 'val_acc':
    max_idx = val_results['val_acc'].idxmax()
    best_epoch = val_results.loc[max_idx, 'epoch']
  
  # Plotting loss
  plt.plot(train_results['epoch'], train_results['train_loss'], label='train_loss')
  plt.plot(val_results['epoch'], val_results['val_loss'], label='val_loss')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title('Loss graph')
  plt.legend()
  plt.savefig('test_graph_loss.png')

  plt.clf()

  # Plotting acc
  plt.plot(train_results['epoch'], train_results['train_acc'], label='train_acc')
  plt.plot(val_results['epoch'], val_results['val_acc'], label='val_acc')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title('Accuracy graph')
  plt.legend()
  plt.savefig('test_graph_acc.png')

  if "test_loss" in log_results.columns:
    test_results = log_results[['test_loss', 'test_acc']].dropna()
    test_loss = test_results['test_loss'].tolist()[0]
    test_acc = test_results['test_acc'].tolist()[0]
  else:
    test_loss = None
    test_acc = None
    
  return test_loss, test_acc, best_epoch

monitor_value = "val_acc"
test_loss, test_acc, best_epoch = plot_results(monitor_value=monitor_value)

print(f"Testing loss: {test_loss}")
print(f"Testing accuracy: {test_acc}")
print(f"Best epoch ({monitor_value}): {best_epoch}")