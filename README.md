# Graph Neural Network Edge Prediction on Cora Dataset

This project implements and compares two approaches for edge prediction on the Cora citation network: Deep Neural Networks (DNN) and Graph Neural Networks (GNN).

## Overview

The Cora dataset consists of 2,708 scientific publications classified into seven classes, with 5,429 citation links between papers. Each publication is described by a 1,433-dimensional binary word vector indicating the presence/absence of dictionary words.

## Project Structure

```
├── gnn.py                           # Main implementation file
├── Cell output for GNN code.pdf     # Training results and outputs
└── README.md                        # This file
```

## Dataset

- **Papers**: 2,708 scientific publications
- **Citations**: 5,429 directed citation links
- **Features**: 1,433-dimensional binary word vectors per paper
- **Classes**: 7 publication categories

## Implementation

### 1. Data Preprocessing

The edge prediction task is formulated as a binary classification problem:
- **Positive samples**: Existing citation edges (5,429 pairs)
- **Negative samples**: Non-existing edges (randomly sampled, equal count)
- **Train/Val/Test split**: 70%/15%/15%

### 2. Deep Neural Network (DNN) Approach

**Feature Engineering for Edges:**
- Concatenation of source and target node features
- Element-wise product (Hadamard product)
- Element-wise absolute difference
- Combined feature dimension: 4,299 (3 × 1,433)

**Architecture:**
- Input layer: 4,299 features
- Hidden layers: [64, 32] units with GELU activation
- Residual blocks with skip connections
- Dropout (0.5) and batch normalization
- Output: Sigmoid activation for binary classification

### 3. Graph Neural Network (GNN) Approach

**Custom GNN Implementation:**
- Message passing with sum aggregation
- Two GNN layers for 2-hop neighborhood information
- Skip connections to prevent over-smoothing
- L2 normalization of node embeddings

**Edge Prediction:**
- Extract learned embeddings for source/target nodes
- Combine embeddings via concatenation, Hadamard product, and L1 distance
- Feed-forward network for final edge scoring

## Results

| Metric | DNN | GNN |
|--------|-----|-----|
| Accuracy | 99.94% | 99.94% |
| ROC AUC | 100.00% | 99.94% |
| PR AUC | 100.00% | 99.94% |

Both models achieved exceptional performance, with the DNN slightly outperforming the GNN on this specific dataset.

## Key Features

### DNN Model
- ✅ Rich feature engineering combining multiple edge representations
- ✅ Residual connections for better gradient flow
- ✅ High performance on feature-rich datasets
- ❌ No explicit graph structure awareness

### GNN Model
- ✅ Natural incorporation of graph topology
- ✅ Message passing for neighborhood aggregation
- ✅ Learnable node embeddings
- ✅ Skip connections to combat over-smoothing
- ❌ More complex implementation

## Training Details

- **Optimizer**: Adam (learning rate: 0.01)
- **Loss**: Binary Cross-Entropy
- **Epochs**: 50
- **Batch Size**: 64
- **Metrics**: Accuracy, AUC

Both models converged quickly with minimal overfitting, achieving near-perfect validation performance within the first few epochs.

## Dependencies

```python
tensorflow>=2.0
pandas
numpy
scikit-learn
matplotlib
```

## Usage

```bash
# Run the complete pipeline
python gnn.py
```

The script will:
1. Download and preprocess the Cora dataset
2. Create train/validation/test splits
3. Train both DNN and GNN models
4. Generate learning curves
5. Evaluate and compare model performance

## Discussion Points

### Performance Analysis
- Both models achieved similar excellent performance (~99.94% accuracy)
- Rich node features (1,433 dimensions) provided sufficient information for DNN
- GNN advantages may be more apparent with sparser feature sets or larger graphs

### Graph Structure Influence
- Cora exhibits homophily (similar papers cite each other)
- Citation network sparsity (density ~0.07%) handled well by both approaches
- GNN naturally captures directional citation relationships

### Scalability Considerations
- Current implementation has O(|V|²) complexity for edge sampling
- Mini-batch training and neighborhood sampling needed for larger graphs
- Distributed training required for very large networks

## Future Improvements

1. **Scalability**: Implement GraphSAGE or FastGCN for large graphs
2. **Sampling**: Smart negative sampling strategies (hard negatives)
3. **Architecture**: Attention mechanisms (GAT) or deeper GNNs
4. **Evaluation**: Test on additional graph datasets with varying characteristics

## References

- Cora Dataset: [Graph Networks Repository](https://graphsandnetworks.com/the-cora-dataset/)
- Original paper citations and methodology details included in code comments
