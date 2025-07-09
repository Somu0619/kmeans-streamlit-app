import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("🧠 K-Means Clustering Explorer")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Select features for clustering
    features = st.multiselect("Select features for clustering", data.columns)

    if len(features) >= 2:
        X = data[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select number of clusters
        k = st.slider("Choose number of clusters (K)", min_value=2, max_value=10, value=3)

        # Fit KMeans
        model = KMeans(n_clusters=k, random_state=42)
        data['Cluster'] = model.fit_predict(X_scaled)

        st.subheader("Clustered Data")
        st.write(data.head())

        # Visualize
        if len(features) == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=data, palette='Set2', ax=ax)
            st.pyplot(fig)
        elif len(features) >= 3:
            st.write("You selected more than 2 features. Displaying pairplot:")
            st.pyplot(sns.pairplot(data[features + ['Cluster']], hue='Cluster', palette='Set2').fig)
    else:
        st.warning("Please select at least 2 features for clustering.")
