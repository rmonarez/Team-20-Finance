
import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# 0. define variables

CSV_PATH = "CleanedUp.csv"  
CRASH_THRESHOLD = -4.0       # define a the crash variable
TEST_SIZE = 0.2              # 20% test, 80% train
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. load and then filter/clean data
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH} in current folder.")

df = pd.read_csv(CSV_PATH)

# filters columns we want to run
cols_to_keep = ["Date", "Index_Change_Percent", "Trading_Volume"]
df = df[cols_to_keep].copy()

# date parsing
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# rows with missing data will be dropped
df = df.dropna(subset=["Index_Change_Percent", "Trading_Volume"])


# 2. CRASH label
# crash = 1 if index change <= threshold, else 0
df["Crash"] = (df["Index_Change_Percent"] <= CRASH_THRESHOLD).astype(int)

print("Total rows:", len(df))
print("Number of crashes:", df["Crash"].sum())
print("Number of non-crashes:", (df["Crash"] == 0).sum())

# 3. plotting the crashes
#= plot: x = date, y = index change, crashes in red
plt.figure(figsize=(10, 5))

normal = df[df["Crash"] == 0]
crash = df[df["Crash"] == 1]

plt.scatter(normal["Date"], normal["Index_Change_Percent"],
            label="Normal", alpha=0.5)
plt.scatter(crash["Date"], crash["Index_Change_Percent"],
            label="Crash", alpha=0.9, marker="x")

plt.axhline(CRASH_THRESHOLD, linestyle="--")
plt.title("Index Change Percent Over Time (Crashes Highlighted)")
plt.xlabel("Date")
plt.ylabel("Index Change Percent")
plt.legend()
plt.tight_layout()
# saves crash plot as png image
plt.savefig("crashes_plot.png", dpi=200)
plt.show()

print("Saved plot as crashes_plot.png")

# 4. preps data for ML model
X = df[["Index_Change_Percent", "Trading_Volume"]].values
y = df["Crash"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# Scaling features (common for ML / neural nets)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. building the neural network
model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)), 
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid") 
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping so it doesn’t overtrain
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 6. training the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# 7. plotting training results

# Loss plot
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
# saves figure
plt.savefig("training_loss.png", dpi=200)
plt.show()

# Accuracy plot
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
# saves figure
plt.savefig("training_accuracy.png", dpi=200)
plt.show()

# 8. evaluate test set
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

