# Toxic Comment Classification using RNN and LSTM

## Project Overview

Deep Learning based **Multi-Label Text Classification** system for detecting different types of toxic content in online comments.

The project explores and compares two recurrent neural network architectures:

* **Bidirectional RNN**
* **LSTM**

The models are trained to identify multiple types of toxicity in the same comment. A single comment can belong to more than one category at the same time.

For example:

```text
Comment:
"You are stupid and disgusting"

Possible labels:
toxic = 1
obscene = 1
insult = 1
```

Therefore, this is a **multi-label classification problem**, rather than a multi-class classification problem.

---

#  Problem Statement

Online platforms receive a huge number of user-generated comments. Manually reviewing all comments for toxic or harmful content is difficult and time-consuming.

The goal of this project is to build a Deep Learning model that can automatically analyze a comment and determine which types of toxic behavior are present.

The system predicts six different toxicity categories:

1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

Each label is treated independently, meaning that multiple labels can be positive for the same comment.

---

# Dataset

The project uses a toxic comment dataset containing text comments and six toxicity labels.

The original training data contains:

```text
159,571 comments
```

After preprocessing and removing invalid/unusable samples:

```text
Before: 159,571
After:  159,078
Removed: 493
```

The dataset contains the following columns:

```text
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

### Dataset Structure

| Column          | Description                    |
| --------------- | ------------------------------ |
| `id`            | Unique comment identifier      |
| `comment_text`  | Original comment text          |
| `toxic`         | General toxic content          |
| `severe_toxic`  | Severe toxic content           |
| `obscene`       | Obscene content                |
| `threat`        | Threatening content            |
| `insult`        | Insulting content              |
| `identity_hate` | Identity-based hateful content |

---

# Label Distribution

The number of positive samples for each label is:

| Label         | Positive Samples |
| ------------- | ---------------: |
| toxic         |           15,294 |
| severe_toxic  |            1,595 |
| obscene       |            8,449 |
| threat        |              478 |
| insult        |            7,877 |
| identity_hate |            1,405 |

This distribution shows a significant **class/label imbalance**.

![Labels](data/data_countable_graph.png)

The `threat` label therefore has far fewer positive examples than `toxic`.

This imbalance becomes an important challenge during model training.

---

# Text Length Analysis

The comment lengths were analyzed before selecting the sequence length used by the models.

The results were:

```text
90%  ≤ 152 words
95%  ≤ 230 words
99%  ≤ 567 words
99.5% ≤ 705 words
Maximum = 1411 words
```

Although some comments are very long, most comments are significantly shorter.

A fixed sequence length was therefore selected for the neural network experiments.

The current experiment uses:

```text
max_length = 100
```

Longer comments are truncated, while shorter comments are padded.

---

# Data Preprocessing

The preprocessing pipeline includes:

* Text normalization
* Tokenization
* Handling URLs
* Handling IP addresses
* Converting text into tokens
* Vocabulary construction
* Converting tokens into integer IDs
* Padding sequences
* Truncating long sequences

For example:

```text
Original:

Congratulations from me as well, use the https://facebook 192.168.1.1
```

After preprocessing:

```text
[
    'congratulations',
    'from',
    'me',
    'as',
    'well',
    'use',
    'the',
    'URL',
    'IP'
]
```

URLs and IP addresses are normalized into special tokens:

```text
URL
IP
```

---

# Vocabulary

A vocabulary was constructed to convert the processed tokens into numerical IDs that can be used by PyTorch.

Special tokens include:

```python
{
    '<PAD>': 0,
    '<UNK>': 1
}
```

Example vocabulary:

```python
{
    '<PAD>': 0,
    '<UNK>': 1,
    'this': 2,
    'is': 3,
    'a': 4,
    'comment': 5,
    'another': 6,
    'you': 7,
    'are': 8,
    'stupid': 9
}
```

### Special Tokens

`<PAD>` is used to make sequences have the same length.

`<UNK>` is used for words that are not found in the vocabulary.

---

#  Input Representation

After preprocessing and vocabulary creation, each comment is represented as a sequence of integer IDs.

For example:

```text
"This is a comment"
```

could become:

```text
[2, 3, 4, 5]
```

The sequence is then padded or truncated to:

```text
max_length = 100
```

The resulting input is passed through an embedding layer.

---

# Model 1 — Bidirectional RNN

The architecture is:

```text
Input Tokens
     ↓
Embedding
     ↓
Bidirectional RNN
     ↓
Last Valid Hidden Representation
     ↓
Fully Connected Layer
     ↓
6 Output Logits
```


### Why the last valid token was selected

Because the sequences are padded, simply taking the last position would sometimes select a `<PAD>` token.

The actual sequence length is therefore calculated:

```python
lengths = (x != 0).sum(dim=1)
```

Then the last real token is selected instead of the padding position.

---

#  Model 2 — BiLSTM

The second architecture explored was an **LSTM**.

The architecture is:

```text
Input Tokens
     ↓
Embedding
     ↓
LSTM
     ↓
Final Hidden State
     ↓
Fully Connected Layer
     ↓
6 Output Logits
```
---

# LSTM Experimental Configuration

The main LSTM experiment used:

| Parameter               |  Value |
| ----------------------- | -----: |
| Embedding Size          |    128 |
| Hidden Dimension        |    256 |
| Number of Layers        |      2 |
| Output Classes          |      6 |
| Epochs                  |     10 |
| Maximum Sequence Length |    100 |
| Batch Size              |     64 |
| Learning Rate           | 0.0001 |
| Dropout                 |    0.3 |

The model uses:

```text
Embedding → LSTM → Fully Connected
```


---

# Handling Label Imbalance

One of the main challenges was the large difference between the number of positive samples for each label.

To address this problem, class weighting was investigated.

The positive weights were calculated using:

```python
positive_counts = y_train[classes].sum()

negative_counts = len(y_train) - positive_counts

pos_weights = np.sqrt(
    negative_counts / positive_counts
)
```

These weights can then be supplied to `BCEWithLogitsLoss`.

The square-root transformation was used to reduce the magnitude of the raw imbalance ratio.

---

# Threshold Experiments

The models produce probabilities between 0 and 1 after applying sigmoid.

A threshold is then used to determine whether a label is positive.

Different thresholds were tested during evaluation.

For example:

```python
predictions = (probabilities >= 0.85).float()
```

and:

```python
predictions = (probabilities >= 0.90).float()
```

The threshold has a direct effect on precision and recall.

A lower threshold generally produces more positive predictions, while a higher threshold makes the model more conservative.

The final threshold should therefore be evaluated together with the F1 scores and per-label performance.

---

# RNN Results

The RNN achieved the following results:

| Label         | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| toxic         |      0.51 |   0.72 | 0.60 |
| severe_toxic  |      0.22 |   0.43 | 0.29 |
| obscene       |      0.56 |   0.63 | 0.59 |
| threat        |      0.30 |   0.32 | 0.31 |
| insult        |      0.51 |   0.56 | 0.54 |
| identity_hate |      0.25 |   0.23 | 0.24 |

The RNN achieved relatively good performance on the more frequent labels such as `toxic` and `obscene`.

Performance was lower on rare labels such as `identity_hate` and `threat`, demonstrating the impact of label imbalance.

![RNN results ](data/RNN_accuracy_results.png)
---

# LSTM Results

The LSTM achieved:

| Label         | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| toxic         |      0.65 |   0.73 | 0.69 |
| severe_toxic  |      0.37 |   0.39 | 0.38 |
| obscene       |      0.69 |   0.68 | 0.69 |
| threat        |      0.36 |   0.58 | 0.45 |
| insult        |      0.69 |   0.56 | 0.62 |
| identity_hate |      0.55 |   0.37 | 0.44 |

![LSTM results ](data/LSTM_accuracy_results.png)
---

# RNN vs LSTM

The main overall results were:

| Metric      |  RNN | LSTM |
| ----------- | ---: | ---: |
| Micro F1    | 0.55 | 0.65 |
| Macro F1    | 0.43 | 0.54 |
| Weighted F1 | 0.55 | 0.65 |

The LSTM experiment produced higher F1 values across the reported overall metrics.


---

# Per-Label Comparison

| Label         | RNN F1 | LSTM F1 |
| ------------- | -----: | ------: |
| toxic         |   0.60 |    0.69 |
| severe_toxic  |   0.29 |    0.38 |
| obscene       |   0.59 |    0.69 |
| threat        |   0.31 |    0.45 |
| insult        |   0.54 |    0.62 |
| identity_hate |   0.24 |    0.44 |

---

# Main Problems Encountered

## 1. Label Imbalance

The dataset contains significantly more examples for some labels than others.

The biggest challenge was the very small number of examples for:

```text
threat
identity_hate
severe_toxic
```

### Solutions investigated

* Class weighting
* Threshold tuning

---

## 2. Padding Problem

The input sequences have different lengths.

Padding was required to create fixed-size tensors.

However, simply selecting the final sequence position can result in selecting a padding token.

### Solution

The RNN calculates the actual sequence length:

```python
lengths = (x != 0).sum(dim=1)
```

and uses the final valid token instead of the final padded position.


---

## 3. Choosing the Evaluation Threshold

Threshold experiments were therefore performed to examine how model predictions change.

---

# Visualizations


## Dataset Distribution

![Label Distribution](data/data_countable_graph.png)


---

# Training Loss

## RNN 

![Training Loss](data/rnn_loss_curve.png)

## LSTM

![Training Loss](data/lstm_loss_curve.png)

# Confusion Matrices


### Severe Toxic

![Severe Toxic Confusion Matrix](data/severe_toxic.png)

### Obscene

![Obscene Confusion Matrix](data/obscene.png)

### Threat

![Threat Confusion Matrix](data/threat.png
)
### Insult
![Insult Confusion Matrix](data/insult.png)