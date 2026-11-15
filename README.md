# LSTM-Based Language Modelling & Text Autocomplete

An LSTM-based next-word prediction model that suggests the most likely next word based on previously typed text. Inspired by Google's Gboard, this project explores building a lightweight language model suitable for fast, on-device text autocomplete.

## Project Overview

The objective is to build a sequential language model that learns conversational patterns and predicts the next word in a sentence using an LSTM architecture.

The model is trained on movie dialogues to capture natural conversational language and provide context-aware word suggestions.

**Live Demo:** https://lstm-autocomplete.streamlit.app/

## Dataset

- **Dataset:** Cornell Movie Dialogues Dataset
- Contains approximately 220K conversational exchanges and 304K utterances.
- Covers dialogues between characters from 617 movies.
- Chosen to capture natural conversational patterns rather than formal written language.

## Data Preprocessing

The preprocessing pipeline involves:

- Extracting dialogue text from raw movie scripts.
- Removing punctuation and digits and converting text to lowercase.
- Tokenizing sentences and building a vocabulary of approximately **64K unique tokens**.
- Converting tokens into integer IDs.
- Generating multiple input-target training samples from each sentence for next-word prediction.
- Applying left zero-padding to maintain a fixed sequence length of 24 tokens.
- Dropping sentences exceeding 24 tokens.

**Final Training Dataset:** Approximately 2.5 million samples.

## Model Architecture

The model uses a single-layer LSTM architecture implemented in PyTorch.

| Layer | Configuration |
|---|---|
| Embedding | 64K vocabulary → 100 dimensions |
| LSTM | 1 layer, 200 hidden units |
| Fully Connected | 200 → 64K output classes |
| Output | Probability distribution over vocabulary |

The final hidden state of the LSTM is passed through a feed-forward layer to predict the next word.

## Model Training

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Epochs | 20 |
| Learning Rate | 0.001 |
| Embedding Dimension | 100 |
| LSTM Hidden Units | 200 |
| Training Samples | ~2.5 Million |
| Hardware | Kaggle T4 GPU |
| Training Time | ~1 hour 40 minutes |

## Results & Observations

The model successfully learns conversational patterns and generates contextually relevant next-word predictions.

### Word Embedding Visualization

The learned 100-dimensional word embeddings were visualized using **t-SNE**.

Words appearing in similar contexts form meaningful clusters. For example, words associated with crime and courtroom scenarios tend to appear close together in the embedding space.

This provides an interesting insight into the linguistic patterns learned by the model.

## Limitations & Future Improvements

- The model expects correctly spelled words; unknown words are all mapped to an `unk` token.
- Predictions are generated after completing a word, not for partially typed words.
- Only the last 10 input tokens are processed during inference to maintain faster predictions.
- The current architecture uses a single-layer LSTM; more advanced architectures and hybrid language modelling techniques could improve prediction quality.

## Tech Stack

- Python
- PyTorch
- Pandas & NumPy
- Matplotlib & Scikit-learn (t-SNE)
- Streamlit

## Links

- **GitHub Repository:** https://github.com/deelipsolanki-DSML/LSTM-language