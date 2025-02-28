pip install numpy pandas seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px

def unsupervised_clustering():
    df = pd.read_csv('dataset.csv')
    scaler = StandardScaler()
    
    # Scale the features
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Only scale numerical data
    
    # Apply K-Means clustering
    unsupervised = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = unsupervised.fit_predict(df_scaled)

    return df

def plot_2d_densityf(df):
    plt.figure(figsize=(8, 6))

    # Filter only Female data
    female_df = df[df['Gender'] == 'Female']

    # KDE plot for Female Age Distribution
    sns.kdeplot(x=female_df["Age"], fill=True, cmap='viridis', thresh=0.05)

    # Labels and title
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Age Distribution of Female Customers")

    # Save the figure
    plt.savefig("static/plot_densityf.png")
    plt.close()

def plot_2d_densitym(df):
    plt.figure(figsize=(8, 6))

    # Filter only Male data
    male_df = df[df['Gender'] == 'Male']

    # KDE plot for Male Age Distribution
    sns.kdeplot(x=male_df["Age"], fill=True, cmap='viridis', thresh=0.05)

    # Labels and title
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Age Distribution of Male Customers")

    # Save the figure
    plt.savefig("static/plot_densitym.png")
    plt.close()

def plot_2d_densityf_1(df):
    plt.figure(figsize=(8, 6))

    # Filter only Female data
    female_df1 = df[df['Gender'] == 'Female']

    # KDE plot for Female count vs. Annual Income
    sns.kdeplot(x=female_df["Annual Income (k$)"], y=female_df["Gender"].value_counts(), fill=True, cmap='viridis', thresh=0.05)

    # Labels and title
    plt.xlabel("Annual Income")
    plt.ylabel("Female Count")
    plt.title("Density Plot: Female Customers vs. Annual Income")

    # Save the figure
    plt.savefig("static/plot_densityf_1.png")
    plt.close()

def plot_2d_densitym_1(df):
    plt.figure(figsize=(8, 6))

    # Filter only Male data
    male_df1 = df[df['Gender'] == 'Male']

    # KDE plot for Male Annual Income
    sns.kdeplot(x=male_df1["Annual Income (k$)"], fill=True, cmap='viridis', thresh=0.05)

    # Labels and title
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Density")
    plt.title("Density Plot: Male Customers' Annual Income")

    # Save the figure
    plt.savefig("static/plot_densitym_1.png")
    plt.close()

plot_2d_densityf(df)
plot_2d_densitym(df)

plot_2d_densityf_1(df)
plot_2d_densitym_1(df)


df = unsupervised_clustering()

