# MLX5 Week 1: Reddit Title Score Predictor Using Word Embeddings

A deep learning model that predicts potential Reddit post scores based on their titles. Built during MLX5 Week 1 as a collaborative project with @kalebsofer and @askarkg12.

## Project Overview

This project explores the relationship between Reddit post titles and their scores using a hybrid neural architecture that combines:
- Word2Vec embeddings for semantic understanding
- Linear layers for score prediction

## Tools & Technologies

- **Deep Learning Framework**: PyTorch
- **Data Processing**: Pandas, PostgreSQL
- **Experiment Tracking**: Weights & Biases (WandB)
- **Development Tools**:
  - poetry for environment + dependency management
  - Git for version control
  - tqdm for progress tracking
  - argparse for CLI argument handling
- **Data Storage**: 
  - PostgreSQL for raw data
  - Parquet for processed datasets

## Technical Implementation

### Core Architecture ([`predictor/model/train.py`](./predictor/model/train.py))

The training pipeline demonstrates several key ML engineering practices. Do note that this is a simplified version of the actual implementation, not optimized for production or security.

1. **Modular Data Handling**
   - Flexible data sourcing (SQL/Parquet)
   - Batch processing capability for large datasets
   - Dynamic sequence padding for variable-length titles

2. **Model Architecture**
   - Pre-trained Word2Vec embeddings (50-dimensional)
   - Configurable hyperparameters via command-line arguments
   - CUDA-ready implementation

3. **Training Infrastructure**
   - WandB integration for experiment tracking
   - Automatic model checkpointing
   - Git commit hash inclusion in saved models

### Key Features

Example of how hyperparameters can be configured:

```python
python train.py --epochs 10 --batch-size 32 --learning-rate 0.001 --seq-length 20
```


- Configurable sequence length for title processing
- Adjustable training window size
- Support for both SQL and Parquet data sources
- Automatic GPU utilization when available

## Development Approach

The implementation prioritizes:
- Modularity and clean code structure
- Robust error handling
- Scalability for large datasets
- Experiment reproducibility

## Setup & Usage

1. Install dependencies
```bash
pip install torch pandas wandb psycopg2 tqdm
```
2. Configure WandB for experiment tracking:
```bash
wandb login
```
3. Run the training script:
```bash
python train.py --epochs 10 --batch-size 32 --learning-rate 0.001 --seq-length 20
```


## Project Structure
```
mlx-word-embeddings/
├── predictor/
│ ├── model/
│ │ ├── train.py # Main training pipeline
│ │ ├── predictor.py # Model architecture
│ │ └── w2v_model.py # Word2Vec implementation
│ └── data/
│   ├── fetch.py # Data loading utilities
│   └── tokenized_titles.parquet
```


## Technical Decisions

- **Word2Vec Integration**: Pre-trained embeddings provide semantic understanding while keeping the model lightweight
- **Batch Processing**: Implemented to handle large-scale Reddit data efficiently
- **Flexible Data Sources**: Support for both SQL and Parquet allows for different deployment scenarios
- **Version Control Integration**: Automatic inclusion of git commits in model checkpoints for full reproducibility

## Future Improvements

- Implement cross-validation
- Add model evaluation metrics
- Expand data preprocessing options
- Add distributed training support

## Acknowledgments

Built as part of the MLX5 program. Special thanks to collaborators and mentors who provided feedback during development.