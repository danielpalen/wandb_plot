import os
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

def load_data(
    api: plt.Axes,
    wandb_project_path: str,
    x_axis: str,
    y_axes: list[str],
    filter: dict,
    fill_nans=True,
    ewm=False,
    every_n_data_points=False,
    use_cache=False,
    cache_dir=os.path.join('.', 'cache'),
    *args, **kwargs
) -> pd.DataFrame:
  """
  This function loads data from wandb and returns it as a dataframe. The purpose is to use it in downstream plotting.
  Therefore, this function uses the concept of x_axis and y_axes to prepare the data. It further computes common
  summary statistics and has options for further manipulations like replacing NaN values.

  :param api: wandb endpoint to load the data from
  :param wandb_project_path: entity and project to load data from
  :param x_axis: x axis which should later be used for plotting
  :param y_axes: list of y axes which can be plotted later
  :param filter: mondoDB style selection filters to query wandb for the desired runs
  :param fill_nans: whether NaN values should be filled with the corresponding y axis mean
  :param ewm: exponentially moving average window size
  :param every_n_data_points: only return every n-th row, i.e. every n-th data point along the x-axis
  :param use_cache: should the function try loading data from local cache
  :param cache_dir:
  :param args:
  :param kwargs:
  :return: dataframe with all data
  """

  hash = hashlib.md5(str({**filter, **{'x': x_axis, 'y': y_axes}}).encode()).hexdigest()
  filters_path = os.path.join(cache_dir, f"{hash}.pkl")

  if use_cache and os.path.isfile(filters_path):
    # already cashed just return from cache
    df = pd.read_pickle(filters_path)
    print(f'Loaded cached data from: {filters_path}')
    return df

  # Load multiple runs
  runs = api.runs(wandb_project_path, filter)

  print(len(runs), "runs loaded for filter", filter)
  if len(runs) == 0:
    print('No Runs found')
    return

  # Aggregate all runs into one Dataframe
  keys = [x_axis] + y_axes

  df = None
  for run in runs:
    if df is None:
        data = run.history(keys=keys)
        if len(data) > 0:
          df = data # Sometimes when the first run that is returned is None, this would fail otherwise.
          first_runs_id = run.id
    else:
        try:
            df = pd.merge(df, run.history(keys=keys), on=x_axis, how='outer', suffixes=(None, f"/{run.id}"))
        except Exception as e:
            print('Error', e)

  if df is None:
    print('WARNING, dataframe is None')
    return

  df = df.sort_values(x_axis)

  # Clean up
  df = df.rename(columns={k:f"{k}/{first_runs_id}" for k in y_axes})
  #df = df.drop(columns=[c for c in df.columns if c.startswith('_step')])
  df = df.replace('Infinity', 1e20)

  # For each y_axis we will now do some more cleanup and aggregation
  for y_axis in y_axes:
    # Column names for all runs for the specified y_axis
    y_axis_runs = [c for c in df.columns if c.startswith(y_axis)]

    # Fill Nans
    if fill_nans:
      df[y_axis_runs] = df[y_axis_runs].T.fillna(df[y_axis_runs].mean(axis=1)).T

    # Exponentially Weighted Moving Average
    if ewm:
      for c in df[y_axis_runs]:
        try:
          df[c] = df[c].ewm(span=ewm, adjust=False).mean()
        except Exception as e:
          print("Failed ewm smoothing", e)

    # if every_n_data_points:
    #   df = df.iloc[::every_n_data_points, :]

    # Statistics
    df[f'{y_axis}/mean'] = df[y_axis_runs].mean(axis=1)
    df[f'{y_axis}/std'] = df[y_axis_runs].std(axis=1)
    df[f'{y_axis}/min'] = df[y_axis_runs].min(axis=1)
    df[f'{y_axis}/max'] = df[y_axis_runs].max(axis=1)

  # save to cache
  os.makedirs(cache_dir, exist_ok=True)
  df.to_pickle(filters_path)
  print(f'Cached data at: {filters_path}')

  return df
