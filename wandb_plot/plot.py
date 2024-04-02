import wandb
import matplotlib.pyplot as plt

import numpy as np

from scipy.ndimage import gaussian_filter1d

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
    smoothing=0,
    every_n_data_points=False,
    color: str=None,
    fill_between: str=None, # min_max or [float]_std (i.e. 2_std)
    label=None,
    custom_fn= lambda x: x,
    alternative_x_axis_fn=None,
    continue_dotted_hline_until=None,
    mean=None,
    iqm=None,
    median=None,
    show_n_runs=False,
    *args, **kwargs
):
  # Load data
  df = load_data(api=api, wandb_project_path=wandb_project_path, x_axis=x_axis, y_axes=[y_axis], filter=filter,
                 fill_nans=fill_nans, every_n_data_points=every_n_data_points, *args, **kwargs)

  if df is None: return

  df = custom_fn(df)

  y_axis_runs = [c for c in df.columns if c.startswith(y_axis) and not (c.split('/')[-1] in ['mean', 'min', 'max', 'std'])]

  percentile = .85
  # df[f"{y_axis}/median"] = df[y_axis_runs].quantile(q=0.5, axis=1)
  df[f"{y_axis}/median"] = df[y_axis_runs].median(axis=1)
  df[f"{y_axis}/quantile_low"] = df[y_axis_runs].quantile(q=1.-percentile, axis=1)
  df[f"{y_axis}/quantile_high"] = df[y_axis_runs].quantile(q=percentile, axis=1)

  # Sanity check that quantiles are in the correct order
  # assert (df[f"{y_axis}/quantile_low"] <= df[f"{y_axis}/quantile_high"]).all()

  # iqm_mask = (df[f"{y_axis}/quantile_low"].values[:,None] <= df[y_axis_runs].values) \
  #   & (df[y_axis_runs].values <= df[f"{y_axis}/quantile_high"].values[:,None])
  
  df[f"{y_axis}/iqm"] = (df[y_axis_runs].quantile(q=0.25, axis=1) + df[y_axis_runs].quantile(q=0.75, axis=1)) / 2

  if every_n_data_points:
    df = df.iloc[::every_n_data_points, :]

  if smoothing:
    for c in ['mean', 'std', 'median', 'quantile_low', 'quantile_high', 'iqm']:
      c_name = f"{y_axis}/{c}"
      first_row_values = df[c_name].iloc[0].copy()
      df[c_name] = gaussian_filter1d(df[c_name], sigma=smoothing)
      # df[c_name].iloc[0] = first_row_values
      df.iloc[0, df.columns.get_loc(c_name)] = first_row_values

  # Plot mean
  if not alternative_x_axis_fn:
    x = df[x_axis]
  else:
    x = alternative_x_axis_fn(df)

  n_runs = len([c for c in df.columns if 'eval/mean' in c and not (c.split('/')[-1] in ['mean', 'min', 'max', 'std', 'median', 'iqm', 'quantile_low', 'quantile_high'])])

  label = f"({n_runs}) {label}" if show_n_runs else label

  if mean:
    p, = axis.plot(x, df[f"{y_axis}/mean"], c=color, label=label, linestyle=kwargs.get('linestyle', None))
  
  if iqm:
    p, = axis.plot(x, df[f"{y_axis}/iqm"], c=color, label=label, linestyle=iqm)

  if median:
    p, = axis.plot(x, df[f"{y_axis}/median"], c=color, label=label, linestyle=median)

  if continue_dotted_hline_until:
    axis.hlines(
      y=df[f"{y_axis}/iqm"].values[-1], 
      xmin=x[-1], 
      xmax=continue_dotted_hline_until, 
      linewidth=1.7, 
      color=color,
      linestyles='--',
      alpha=0.5
    )

  

  # plot shaded std
  if fill_between:
    color = p.get_color() if color is None else color

    if fill_between == 'min_max':
      axis.fill_between(x, df[f"{y_axis}/min"], df[f"{y_axis}/max"], alpha=0.2, color=color)

    elif fill_between.endswith('_quantile'):
      axis.fill_between(x, df[f"{y_axis}/quantile_low"], df[f"{y_axis}/quantile_high"], alpha=0.2, color=color)

    else:
      a = float(fill_between.split('_')[0])
      min = df[f"{y_axis}/mean"] - a * df[f"{y_axis}/std"]
      max = df[f"{y_axis}/mean"] + a * df[f"{y_axis}/std"]
      axis.fill_between(x, min, max, alpha=0.2, color=color)
