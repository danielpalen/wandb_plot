import wandb
import matplotlib.pyplot as plt

from wandb_plot.data_loader import load_data

def plot_wandb(
    axis: plt.Axes,
    api: wandb.Api,
    wandb_project_path: str,
    x_axis: str,
    y_axis: str,
    filter: dict,
    fill_nans=True,
    ewm=False,
    every_n_data_points=False,
    color: str=None,
    fill_between: str=None, # min_max or [float]_std (i.e. 2_std)
    label=None,
    custom_fn= lambda x: x,
    *args, **kwargs
):
  # Load data
  df = load_data(api=api, wandb_project_path=wandb_project_path, x_axis=x_axis, y_axes=[y_axis], filter=filter,
                 fill_nans=fill_nans, ewm=ewm, every_n_data_points=every_n_data_points, *args, **kwargs)

  if df is None: return

  df = custom_fn(df)

  if every_n_data_points:
    df = df.iloc[::every_n_data_points, :]

  # Plot mean
  p, = axis.plot(df[x_axis], df[f"{y_axis}/mean"], c=color, label=label)

  # plot shaded std
  if fill_between:
    color = p.get_color() if color is None else color

    if fill_between == 'min_max':
      axis.fill_between(df[x_axis], df[f"{y_axis}/min"], df[f"{y_axis}/max"], alpha=0.2, color=color)

    else:
      a = float(fill_between.split('_')[0])
      min = df[f"{y_axis}/mean"] - a * df[f"{y_axis}/std"]
      max = df[f"{y_axis}/mean"] + a * df[f"{y_axis}/std"]
      axis.fill_between(df[x_axis], min, max, alpha=0.2, color=color)
