import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("haberman.csv")

def main():

    print ("iris dataset shape : ",df.shape)
    print ("all the columns:\n",df.columns)
    print("species distribution",df["status"].value_counts())
    print("not a fully balanced dataset\n")

    # replacing 1 and 2 with yes and no
    df['status'] = df['status'].map({1:"yes", 2:"no"})



    print("pair plot of iris dataset:\n")
    sns.set_style("whitegrid")
    sns.pairplot(df, hue="status", height=2, diag_kind='hist')
    plt.show()
    print("\ndata Description:\n")
    print(df.describe())



if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main()