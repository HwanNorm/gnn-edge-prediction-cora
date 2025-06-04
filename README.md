# Graph Neural Network for Edge Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.94%25-green.svg)

## 🎯 Project Overview

This project compares two different approaches for predicting connections in networks. I built both a regular neural network and a graph neural network to predict whether two research papers should cite each other.

## 📊 Key Results

| Model | Accuracy | ROC AUC | PR AUC | Training Time |
|-------|----------|---------|---------|---------------|
| **DNN** | 99.94% | 100.00% | 100.00% | ~22s initial + ~1s/epoch |
| **GNN** | 99.94% | 99.94% | 99.94% | ~20s initial + ~1s/epoch |

Both models achieved exceptional performance, demonstrating the effectiveness of the implemented architectures for network analysis tasks.

## 🚀 Features

### Core Implementation
- **Custom GNN Architecture**: Built from scratch with message passing and aggregation layers
- **Comparative Analysis**: Systematic comparison between DNN and GNN approaches
- **Advanced Feature Engineering**: Multiple edge representation techniques (concatenation, Hadamard product, L1 distance)
- **Robust Evaluation**: Comprehensive metrics including accuracy, ROC AUC, and PR AUC

### Technical Highlights
- **Message Passing Framework**: Custom implementation of graph convolution with neighbor aggregation
- **Scalable Architecture**: Efficient TensorFlow implementation with batch processing
- **Skip Connections**: Residual connections to prevent over-smoothing in deep GNN layers
- **Multi-Modal Edge Features**: Combined embeddings using concatenation, element-wise operations

## 🏗️ How It Works

### Graph Neural Network Approach
The GNN works by having each paper "talk" to papers it cites and learn from them:

1. **Each paper starts with its own features** (words that appear in the paper)
2. **Papers share information** with papers they cite or are cited by  
3. **Papers update their understanding** based on what they learned from neighbors
4. **This happens multiple times** to capture complex relationships
5. **Finally, we predict** if two papers should be connected based on their updated features

### Why This Is Useful
- Regular neural networks only look at individual papers in isolation
- Graph neural networks can learn from the network structure itself
- This often gives better results for problems involving relationships between things

## 📈 Results

### What I Found
- **Both approaches worked very well**: 99.94% accuracy for both models
- **Fast learning**: Both models learned quickly in the first few training rounds
- **Stable training**: Performance stayed consistent throughout training
- **Similar results**: Both the regular neural network and graph neural network performed almost equally well

### Interesting Observations
- The regular neural network was slightly faster to train
- The graph neural network took a bit longer but used the network structure naturally
- Both models were good at not overfitting (memorizing training data)

## 🔬 How I Set Up The Experiment

### Creating the Training Data
- **Positive examples**: Used real citation links that exist in the dataset (2,714 examples)
- **Negative examples**: Randomly picked paper pairs that don't cite each other (2,714 examples)
- **Total dataset**: 7,600 examples split into training, validation, and test sets

### Features I Used
For each pair of papers, I combined their information in three ways:
- **Concatenation**: Put both papers' features side by side
- **Element-wise multiplication**: Multiply corresponding features together  
- **Absolute difference**: Find the difference between corresponding features

This gave each paper pair a rich set of features to learn from.

## 💼 What This Project Actually Does

**Current Implementation:**
- Predicts citation links between academic papers
- Uses a dataset of 2,708 computer science research papers
- Learns which papers should cite each other based on their content and existing citation patterns

**Potential Applications (with modifications):**
The same techniques could potentially be adapted for:
- **Social Networks**: Predicting friendships or connections between users
- **E-commerce**: Recommending products based on user-item networks  
- **Citation Analysis**: Academic research and knowledge mapping
- **General Network Analysis**: Any problem involving predicting connections between entities

**Note**: This project demonstrates the core techniques, but would need significant changes and new training data to work for other domains like banking or security.

## 🛠️ Technical Implementation

## 🚀 How to Run This Project

### Requirements
```bash
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GNN-Edge-Prediction.git
   cd GNN-Edge-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow numpy pandas scikit-learn matplotlib
   ```

### Usage
Simply open and run the Jupyter notebook:
```bash
jupyter notebook GNN_Cora.ipynb
```

The notebook will:
- Automatically download the Cora dataset
- Train both DNN and GNN models
- Display training progress and results
- Show performance comparisons

**Expected runtime**: About 5-10 minutes on a regular laptop (longer on first run due to dataset download)

## 📁 Project Structure
```
GNN-Edge-Prediction/
├── GNN_Cora.ipynb          # Main notebook with all code and analysis
├── README.md               # This file
└── requirements.txt        # Python dependencies (optional)
```

### Model Architecture Details
- **Hidden Units**: [64, 32] for both DNN and GNN
- **Dropout Rate**: 0.5 for regularization
- **Optimizer**: Adam with 0.01 learning rate
- **Loss Function**: Binary cross-entropy
- **Activation**: GELU for hidden layers, sigmoid for output

### Key Implementation Features
```python
# Custom aggregation function
def aggregate(self, neighbor_messages, node_indices, num_nodes):
    return tf.math.unsorted_segment_sum(
        neighbor_messages, node_indices, num_segments=num_nodes
    )

# Multi-faceted edge representation
concat_embeddings = tf.concat([source_embeddings, target_embeddings], axis=1)
hadamard_embeddings = source_embeddings * target_embeddings  
l1_embeddings = tf.abs(source_embeddings - target_embeddings)
combined_embeddings = tf.concat([concat_embeddings, hadamard_embeddings, l1_embeddings], axis=1)
```

## 📊 How I Measured Success

### Metrics Used
- **Accuracy**: How often the model correctly predicted whether papers should cite each other (99.94%)
- **ROC AUC**: How well the model separates positive and negative examples (near perfect)
- **PR AUC**: How precise the model is when it predicts a citation should exist (near perfect)

### Why These Results Are Good
- Both models learned very quickly and reached high accuracy
- The validation performance was similar to training performance (no overfitting)
- Results were consistent across different runs

## 🔮 Possible Improvements

### For Better Performance
- Try different neural network architectures (deeper networks, attention mechanisms)
- Test with other graph datasets to see how well the approach generalizes
- Experiment with different ways of combining node features

### For Larger Datasets
- Implement mini-batch training for datasets that don't fit in memory
- Add support for different types of relationships between nodes
- Test on networks with millions of nodes and edges

## 📝 The Dataset I Used

I worked with the **Cora citation dataset**, which contains:
- **2,708 computer science research papers** on machine learning topics
- **5,429 citation links** (when one paper references another)
- **1,433 unique words** that describe each paper's content

**The Task**: Given two papers, predict whether one should cite the other based on:
- The words that appear in each paper
- The existing citation patterns in the network
- The relationships between papers and their topics

This is a well-known dataset used to test graph learning algorithms.

## 🎯 What I Learned

This project helped me understand:
- How to build neural networks that work with graph data (not just regular data)
- The difference between traditional neural networks and graph-based approaches
- How to properly evaluate machine learning models with multiple metrics
- How to implement complex algorithms from research papers using TensorFlow

The techniques I learned here could be useful for any problem involving networks or relationships between data points.
