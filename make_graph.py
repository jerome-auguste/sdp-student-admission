import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# df = pd.read_csv('results.csv') # run 'generate_csv.py' to make it (c long)

# adapt for each graph
# a faire en live pour les besoins du rapports
# on a tout ce qui faut dans results.csv normalement

def reload():
    plt.clf()
    df = pd.read_csv('results.csv')
    for col in df.columns:
        if 'accuracy' in col or 'noise' in col:
            df[col] = df[col].apply(lambda acc: acc*100)
    return df

def temps_fct_size():
    df = reload()
    df = df[(df["num_criteria"]==4) & (df["num_classes"]==2)]
    df = df.groupby(['name', 'size']).mean().reset_index()
    df = df.pivot('size', 'name', 'time')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(yscale='log', ylabel='time (s)', xlabel="Taille du learning set")
    axes.set_title('Temps de calcul')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('temps_fct_size.jpg')
    plt.show()

def temps_fct_criteria():
    df = reload()
    df = df[(df["num_classes"]==2) & (df['size']==50)]
    df = df.groupby(['name', 'num_criteria']).mean().reset_index()
    df = df.pivot('num_criteria', 'name', 'time')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(yscale='log', ylabel='time (s)', xlabel="Nombre de critere")
    axes.set_title('Temps de calcul')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('temps_fct_criteria.jpg')
    plt.show()


def accuracy_train_fct_size():
    df = reload()
    # df = df[(df["num_criteria"]==4) & (df["num_classes"]==2)]
    df = df[(df["noise"]==0)]
    df = df.groupby(['name', 'size']).mean().reset_index()
    df = df.pivot('size', 'name', 'accuracy_on_train')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(ylabel='Accuracy (%)', xlabel="Taille du learning set")
    axes.set_title('Proportion de bonne reproduction sur le learning set')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('accuracy_train.jpg')
    plt.show()

def accuracy_test_fct_size():
    df = reload()
    df = df[(df["noise"]==0)]
    # df = df[(df["num_criteria"]==4) & (df["num_classes"]==2)]
    df = df.groupby(['name', 'size']).mean().reset_index()
    df = df.pivot('size', 'name', 'accuracy_on_test')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(ylabel='Accuracy (%)', xlabel="Taille du learning set")
    axes.set_title('Accuracy (test set)')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('accuracy_test.jpg')
    plt.show()

def accuracy_test_fct_noise():
    df = reload()
    # df = df[(df["num_criteria"]==4) & (df["num_classes"]==2)]
    df = df.groupby(['name', 'noise']).mean().reset_index()
    df = df.pivot('noise', 'name', 'accuracy_on_test')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(ylabel='Accuracy (%)', xlabel="Proportion de données bruitées (%)")
    axes.set_title('Accuracy en fonction du bruit')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('accuracy_test_noise.jpg')
    plt.show()


def accuracy_test_fct_criteria():
    df = reload()
    df = df[(df["noise"]==0)]
    df = df.groupby(['name', 'num_criteria']).mean().reset_index()
    df = df.pivot('num_criteria', 'name', 'accuracy_on_test')

    axes = sns.lineplot(data=df, markers=['o']*len(df))
    axes.set(ylabel='Accuracy (%)', xlabel="Taille du learning set")
    axes.set_title('Accuracy en fonction du nombre de critères')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig('accuracy_test_criteria.jpg')
    plt.show()



temps_fct_size()
temps_fct_criteria()
accuracy_train_fct_size()
accuracy_test_fct_size()
accuracy_test_fct_noise()
accuracy_test_fct_criteria()
