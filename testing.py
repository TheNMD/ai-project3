import pandas as pd
import matplotlib.pyplot as plt

result_path = "result"

def plot_results(model_name, model_option, latest_version):
  log_results = pd.read_csv(f"{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/metrics.csv")
  train_results = log_results[['epoch', 'train_loss', 'train_acc']].dropna()
  train_results = train_results.groupby(['epoch'], as_index=False).mean()
  val_results = log_results[['epoch', 'val_loss', 'val_acc']].dropna()
  val_results = val_results.groupby(['epoch'], as_index=False).mean()

  # Plotting loss
  plt.plot(train_results['epoch'], train_results['train_loss'], label='train_loss')
  plt.plot(val_results['epoch'], val_results['val_loss'], label='val_loss')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title(f'Loss of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_loss.png')

  plt.clf()

  # Plotting acc
  plt.plot(train_results['epoch'], train_results['train_acc'], label='train_acc')
  plt.plot(val_results['epoch'], val_results['val_acc'], label='val_acc')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title(f'Accuracy of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_acc.png')

  test_results = log_results[['test_loss', 'test_acc']].dropna()
  test_loss = test_results['test_loss'].tolist()[0]
  test_acc = test_results['test_acc'].tolist()[0]
  print(f"Testing loss: {test_loss}")
  print(f"Testing acc: {test_acc}")
  
  return test_loss, test_acc

model_name = "convnext-b"
model_option = "pretrained"
latest_version = "version_0"

plot_results(model_name, model_option, latest_version)