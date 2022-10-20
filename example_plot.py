
########################################
# Generate some fake data
########################################
import wandb
import numpy as np

# we will create
for n in range(3):
  wandb.init(entity='ias', project='wandb_plot', name=f'run_{n}', group='test_runs')
  for t in range(10):
    wandb.log({'x': t, 'y': np.random.randn()})
  wandb.finish()


########################################
# Retreive and Plotting Data
########################################
import matplotlib.pyplot as plt
from wandb_plot.plot import plot_wandb

fig, ax = plt.subplots(1, 1)

wandb_config = dict(api=wandb.Api(), wandb_project_path='ias/wandb_plot')
plt_config = dict(fill_between='min_max')

plot_wandb(
  axis=ax, x_axis='x', y_axis='y',
  filter={'group': 'test_runs'},
  color='tab:blue', label='Test Data straight from WandB',
  **wandb_config, **plt_config,
)

plt.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=.13)
fig.legend(loc='lower center', framealpha=0.0)
plt.show()