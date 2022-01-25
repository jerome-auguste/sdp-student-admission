import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('results.csv') # run 'generate_csv.py' to make it (c long)

# adapt for each graph
# a faire en live pour les besoins du rapports
# on a tout ce qui faut dans results.csv normalement
df = df[(df["num_criteria"]==4) & (df["num_classes"]==2)]
df = df.groupby(['name', 'size']).mean().reset_index()
df = df.pivot('size', 'name', 'time')

axes = sns.lineplot(data=df, markers=['o', 'o'])
axes.set(yscale='log', ylabel='time (s)')

# plt.show()

fig = axes.get_figure()
fig.savefig('nice_fig.jpg')