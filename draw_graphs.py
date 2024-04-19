import pandas as pd
import matplotlib.pyplot as plt

def plot_results(monitor_value="val_acc"):
  log_results = pd.read_csv("metrics.csv")
  train_results = log_results[['epoch', 'train_loss', 'train_acc']].dropna()
  train_results = train_results.groupby(['epoch'], as_index=False).mean()
  val_results = log_results[['epoch', 'val_loss', 'val_acc']].dropna()
  val_results = val_results.groupby(['epoch'], as_index=False).mean()

  if monitor_value == 'val_loss':
    best_epoch = val_results['val_loss'].idxmin()
  elif monitor_value == 'val_acc':
    best_epoch = val_results['val_acc'].idxmax()
  
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

  test_results = log_results[['test_loss', 'test_acc']].dropna()
  test_loss = test_results['test_loss'].tolist()[0]
  test_acc = test_results['test_acc'].tolist()[0]
  
  return test_loss, test_acc, best_epoch

test_loss, test_acc, best_epoch = plot_results(monitor_value="val_acc")

print(f"Testing loss: {test_loss}")
print(f"Testing accuray: {test_acc}")
print(f"Best epoch: {best_epoch}")