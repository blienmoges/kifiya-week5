
import matplotlib.pyplot as plt

def plot_class_distribution(y, title="Class Distribution"):
    y.value_counts().plot(kind='bar', title=title)
    plt.show()

def plot_histogram(df, column, bins=50, title=None):
    df[column].hist(bins=bins)
    plt.title(title or column)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()
