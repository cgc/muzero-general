import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = 'results/gridworld/2022-03-29--16-59-22'                                                                                                                            

event_accumulator = EventAccumulator(log_dir)
event_accumulator.Reload()

keys = [
    '1.Total_reward/1.Total_reward',
    '1.Total_reward/3.Episode_length',
    '2.Workers/1.Self_played_games',
    '2.Workers/3.Self_played_steps',
]

# making sure all steps are the same
steps = [e.step for e in event_accumulator.Scalars(keys[0])]

df = {}

for key in keys:
    key_steps = [e.step for e in event_accumulator.Scalars(key)]
    assert key_steps == steps, key
    df[key.split('.')[-1]] = [e.value for e in event_accumulator.Scalars(key)]

df = pd.DataFrame(df)
df = df.ewm(alpha=0.1).mean()
print(df.shape)
#df.to_csv("exported_run.csv")
#print(df)

try:
    os.makedirs('figures')
except: pass

def make_fig(x, y):
    plt.figure()
    sns.lineplot(x=x, y=y, data=df)
    plt.savefig(f'figures/{y}_per_{x}.png')
    plt.close()

for x in ['Self_played_games', 'Self_played_steps']:
    for y in ['Total_reward', 'Episode_length']:
        make_fig(x, y)
