"""
Spam Email Classifier using Neural Networks

A capstone project following Andrew Ng's ML workflow:
1. Load and prepare dataset
2. Train on 60% of data
3. Cross-validate on 20% to detect overfitting
4. Final evaluation on remaining 20%
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping                                                                                                     
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# STEP 1: Load and Explore the Dataset
# =============================================================================

# Define column names based on spambase.names documentation
feature_names = [
    # 48 word frequency features
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    # 6 character frequency features
    'char_freq_semicolon', 'char_freq_paren', 'char_freq_bracket',
    'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
    # 3 capital letter features
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
    # Target
    'is_spam'
]

# Load the dataset
df = pd.read_csv('data/spambase.data', header=None, names=feature_names)

#   The data exploration showed us:                                                                                                               
#  - 4601 samples, 57 features, binary target (spam/not spam)                                                                                    
#  - No missing values (lucky - no imputation needed)                                                                                            
#  - Class ratio is 39% spam / 61% not spam (reasonably balanced)  


# =============================================================================
# STEP 2: Split the Dataset
# =============================================================================

X = df.drop('is_spam', axis=1).values
y = df['is_spam'].values

# First split: 60% train, 40% temp to split into val/test                                                                                                          
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)                                                                 
                                                                                                                                            
# Second split: 50% of temp = 20% val, 20% test                                                                                               
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Scaling features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 3: Build and Train the Neural Network
# =============================================================================
model = keras.Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# val_loss needs to be monitored as model tends to overfit quickly
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])

# Test model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the trained model
model.save('model.keras')

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')