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

# 1. configure variables
CSV_PATH = "sensex.csv"      # path to your CSV
DATE_COL = "Price"            # change if your column is named differently
CLOSE_COL = "Close"          # change if needed

CRASH_THRESHOLD_DAILY = -5   # % drop to call a "crash day"
CRASH_HORIZON = 10           # look-ahead window (days) for future crash label
SEQ_LEN = 60                 # how many past days the model sees


# ML run throughs
TEST_SIZE = 0.2
EPOCHS = 5          # fewer passes over the data
BATCH_SIZE = 128    # bigger batches = fewer
LEARNING_RATE = 1e-3

# 2. load data
# csv to be changed to whatever data we are pulling from
print("Working directory:", os.getcwd())
df = pd.read_csv(CSV_PATH)
print("\nColumns in CSV:", df.columns.tolist())
print("\nRaw head:")
print(df.head())

# if CSV has a bad header row, fix it
# This removes that row and resets index
if str(df[DATE_COL].iloc[0]).strip().lower() == "date":
    df = df.iloc[1:].copy()

df.reset_index(drop=True, inplace=True)


if DATE_COL not in df.columns or CLOSE_COL not in df.columns:
    raise ValueError(f"Make sure '{DATE_COL}' and '{CLOSE_COL}' exist in your CSV.")

# 3. process data 
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL)
df.set_index(DATE_COL, inplace=True)

print("\nAfter date processing:")
print(df[[CLOSE_COL]].head())

# Daily return (%)
df["Daily_Return"] = df[CLOSE_COL].pct_change() * 100

# Drop first row with NaN daily return
df = df.dropna(subset=["Daily_Return"]).copy()

# Flags daily crash
df["Crash_daily"] = df["Daily_Return"] <= CRASH_THRESHOLD_DAILY

print("\nCrash_daily counts:")
print(df["Crash_daily"].value_counts())

# 4. plot price with the crash days using matplotlib 

plt.figure(figsize=(14, 6))
plt.plot(df.index, df[CLOSE_COL], label="Sensex Close", color="blue")

crash_days = df.index[df["Crash_daily"]]
crash_closes = df.loc[df["Crash_daily"], CLOSE_COL]

plt.scatter(crash_days, crash_closes,
            color="red", label=f"Daily Crash (≤ {CRASH_THRESHOLD_DAILY}%)",
            zorder=5)

plt.title("Sensex Closing Price with Daily Crashes Highlighted")
plt.xlabel("Date")
plt.ylabel("Sensex Close")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 5. builds future crash for ML
#    Crash_future = at least one daily crash in the next CRASH_HORIZON days
n = len(df)
crash_future = np.zeros(n, dtype=bool)

returns = df["Daily_Return"].values

for i in range(n - CRASH_HORIZON):
    future_window = returns[i+1 : i+1+CRASH_HORIZON]
    crash_future[i] = (future_window <= CRASH_THRESHOLD_DAILY).any()

df["Crash_future"] = crash_future

# Drop last CRASH_HORIZON rows since they don't have full look-ahead
df = df.iloc[: n - CRASH_HORIZON].copy()

print("\nCrash_future counts:")
print(df["Crash_future"].value_counts())

# 6. create sequences for LSTM 

feature_cols = [CLOSE_COL, "Daily_Return"]
values = df[feature_cols].values.astype(np.float32)
labels = df["Crash_future"].astype(int).values

X = []
y = []

for i in range(SEQ_LEN, len(df)):
    X.append(values[i-SEQ_LEN:i])  # previous SEQ_LEN days
    y.append(labels[i])            # label for today

X = np.array(X)
y = np.array(y)

print("\nSequence data shapes:")
print("X:", X.shape, "y:", y.shape)
print("Crash_future distribution:", np.bincount(y))

if X.shape[0] == 0:
    raise ValueError("Not enough data for the chosen SEQ_LEN. Try a smaller SEQ_LEN.")

# 7. train and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

num_train, seq_len, num_features = X_train.shape
print(f"\nTrain samples: {num_train}, Test samples: {X_test.shape[0]}")
print(f"Sequence length: {seq_len}, Num features: {num_features}")

# Flatten for scaler
X_train_2d = X_train.reshape(-1, num_features)
X_test_2d = X_test.reshape(-1, num_features)

scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_test_scaled_2d = scaler.transform(X_test_2d)

X_train_scaled = X_train_scaled_2d.reshape(-1, seq_len, num_features)
X_test_scaled = X_test_scaled_2d.reshape(-1, seq_len, num_features)

# 8. build the LSTM model

model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, num_features)),
    layers.LSTM(32),              # single smaller LSTM layer
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)

model.summary()

# 9. train the model

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)



# 10. model evaluation

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# 10.5 display dates and predictions for crashes

# Each X sample corresponds to df index from SEQ_LEN onwards
all_dates = df.index[SEQ_LEN:]                 # aligned with X / y
test_dates = all_dates[-len(X_test):]          # last part = test set

results = pd.DataFrame({
    "Date": test_dates,
    "TrueCrashFuture": y_test,
    "PredCrashProb": y_pred_proba
})
results["PredClass"] = (results["PredCrashProb"] >= 0.5).astype(int)

# Show the last 10 test dates (most recent)
print("\n=== Recent predictions (chronological, last 10 test days) ===")
print(results.tail(10))

# Show the 10 dates with highest predicted crash probability
print("\n=== Top 10 highest predicted crash probabilities ===")
print(
    results.sort_values("PredCrashProb", ascending=False).head(10)
)



# 11. plot for the ML on the training curves

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
