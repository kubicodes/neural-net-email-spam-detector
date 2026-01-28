# Spam Email Classifier

A neural network that classifies emails as spam or not spam. Built this as a capstone project after completing Andrew Ng's Machine Learning Specialization to get hands-on experience with the full ML workflow.

## Dataset

Using the [Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase) from UCI Machine Learning Repository. It contains 4601 emails, each represented as 57 features:

- Word frequencies (how often words like "free", "money", "credit" appear)
- Character frequencies (!, $, etc.)
- Capital letter statistics

The target is binary: 1 = spam, 0 = not spam. The dataset is roughly 39% spam, 61% not spam.

## Workflow

**1. Data preparation**
- Split into 60% training, 20% validation, 20% test (stratified to maintain spam ratio)
- Scale features using StandardScaler (fit on training data only to avoid data leakage)

**2. Model**
- Input layer: 57 features
- Hidden layer 1: 32 neurons, ReLU activation
- Hidden layer 2: 16 neurons, ReLU activation
- Output layer: 1 neuron, sigmoid activation (outputs spam probability)

**3. Training**
- Optimizer: Adam
- Loss: Binary crossentropy
- Early stopping on validation loss (patience=5) to prevent overfitting

**4. Evaluation**
- Final test accuracy: ~93%

## Training output

The model trains for about 10-15 epochs before early stopping kicks in. You'll see something like:

```
Epoch 1/100 - accuracy: 0.72 - loss: 0.56 - val_accuracy: 0.88 - val_loss: 0.36
Epoch 2/100 - accuracy: 0.89 - loss: 0.31 - val_accuracy: 0.91 - val_loss: 0.25
...
Epoch 9/100 - accuracy: 0.95 - loss: 0.14 - val_accuracy: 0.93 - val_loss: 0.18
```

Training stops when validation loss stops improving. The gap between training accuracy (~95%) and validation accuracy (~93%) stays small, indicating the model generalizes well.

## Learnings

The first training run showed clear overfitting. Training accuracy kept climbing to 98% while validation accuracy plateaued around 93%. More telling was the validation loss - it dropped to ~0.17 around epoch 13, then started rising again while training loss kept decreasing. Classic overfitting.

Tried a few things to fix it:

**Smaller model (32-16 → 16-8 neurons):** Helped a bit - the overfitting was less aggressive, but still there. Validation loss still crept up after epoch 17.

**L2 regularization:** Added weight penalties to discourage the model from relying too heavily on specific features. Didn't really improve things, and the overall validation loss was actually slightly worse.

**What actually worked - early stopping:** Instead of trying to prevent the model from overfitting, just stop training when it starts. Monitor validation loss, and if it doesn't improve for 5 epochs, stop and keep the best weights. Simple and effective.

The takeaway: the model without any tricks achieved its best validation performance around epoch 12-15. All the regularization attempts were just trying to prevent training past that point. Early stopping does this directly without needing to tune regularization strength or layer sizes.

## Usage

```bash
python spam_classifier.py
```

This trains the model and saves `model.keras` and `scaler.pkl` for later use.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Possible improvements

- Build a text preprocessor that takes raw email content and extracts the 57 features, so you can classify actual emails instead of pre-processed data
- Try different architectures (more/fewer layers, different neuron counts)
- Experiment with dropout regularization
- Use a more modern approach with word embeddings instead of manual feature engineering
