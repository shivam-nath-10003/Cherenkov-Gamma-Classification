import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
import streamlit as st

# Load the dataset
cols = ["fLength", "fWidth", "fSize", "sCoc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
dp = pd.read_csv("/Users/shivamnath/Downloads/magic+gamma+telescope/magic04.data", names=cols)

# Convert class labels to 0 and 1
dp["class"] = (dp["class"] == "g").astype(int)

        
def display_histograms(dataframe, columns):
    for label in columns:
        fig, ax = plt.subplots()  # Create a new figure
        ax.hist(dataframe[dataframe["class"] == 1][label], color='red', label='gamma', alpha=0.7, density=True)
        ax.hist(dataframe[dataframe["class"] == 0][label], color='yellow', label='hadron', alpha=0.7, density=True)
        ax.set_title(label)
        ax.set_ylabel("Probability")
        ax.set_xlabel(label)
        ax.legend()
        st.pyplot(fig)  # Pass the figure to st.pyplot()

# Function to scale dataset and perform oversampling
def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    return x, y

# Function to train machine learning models
def train_models(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # kNN (K Nearest Neighbors)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    knn_accuracy = knn_model.score(x_test, y_test)
    st.write("kNN Accuracy:", knn_accuracy)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    nb_accuracy = nb_model.score(x_test, y_test)
    st.write("Naive Bayes Accuracy:", nb_accuracy)

    # Logistic Regression
    lg_model = LogisticRegression()
    lg_model.fit(x_train, y_train)
    lg_accuracy = lg_model.score(x_test, y_test)
    st.write("Logistic Regression Accuracy:", lg_accuracy)

    # SVM (Support Vector Machines)
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    svm_accuracy = svm_model.score(x_test, y_test)
    st.write("SVM Accuracy:", svm_accuracy)

    # Neural Net
    def train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs):
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation='relu'),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        history = nn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        return nn_model, history

    least_val_loss = float('inf')
    least_loss_model = None

    epochs = 100
    for num_nodes in [64, 128]:
        for dropout_prob in [0, 0.2]:
            for learning_rate in [0.001, 0.005]:
                for batch_size in [32, 64]:
                    model, history = train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs)
                    val_loss = model.evaluate(x_valid, y_valid)[0]
                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = model

    nn_accuracy = least_loss_model.evaluate(x_test, y_test)[1]
    st.write("Neural Net Accuracy:", nn_accuracy)

# Create Streamlit app
def main():
    st.title('Machine Learning Models')

    # Display histograms
    st.header('Histograms')
    display_histograms(dp, cols[:-1])  # Exclude the last column (class) from histograms

    # Scale dataset and perform oversampling
    st.header('Data Scaling')
    oversample = st.checkbox("Perform Oversampling")
    x_train, y_train = scale_dataset(dp, oversample)
    x_valid, y_valid = scale_dataset(dp, False)
    x_test, y_test = scale_dataset(dp, False)

    # Train machine learning models
    st.header('Training Models')
    train_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == '__main__':
    main()
