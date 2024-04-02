import wandb
from tqdm import tqdm
from pprint import pprint

api = wandb.Api()

runs = api.runs('ias/model_based_rl')

meta = {}

for run in tqdm(runs):
    files = run.files()
    pkl_files = [f for f in files if '.pkl' in f.name]
    # if pkl_files:
    #     meta[files.variables['name']] = len(pkl_files)
    for f in pkl_files:
        f.delete()

pprint(meta)
