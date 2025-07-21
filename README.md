# CarbonIQ-2.0

## Overview
CarbonIQ-2.0 is an advanced machine learning system that uses Graph Neural Networks (GNNs) and Multi-Armed Bandit algorithms to optimize supplier selection for carbon efficiency in supply chains. The system analyzes supplier relationships, predicts carbon flows between entities, and dynamically selects optimal suppliers based on multiple factors including carbon intensity, performance, and cost efficiency.

## Features
- **Graph Neural Network (GNN) Model**: Analyzes complex supplier relationships and predicts carbon flows
- **Multi-Armed Bandit Integration**: Optimizes supplier selection through reinforcement learning
- **Dynamic Supplier Selection**: Adapts to changing demand and supply conditions
- **Multi-Task Learning**: Predicts supplier classifications, locations, and performance metrics
- **Real-Time Inference API**: Provides carbon flow predictions and supplier embeddings via FastAPI

## Architecture
The system consists of several key components:
- **ImprovedCarbonGNN**: Advanced GNN architecture with attention mechanisms and residual connections
- **SupplierBanditAgent**: Reinforcement learning agent for optimal supplier selection
- **GNNBanditIntegration**: Integration layer connecting GNN insights with bandit decision-making
- **CarbonGraphBuilder**: Constructs supplier relationship graphs from raw data
- **EnhancedGNNTrainer**: Training pipeline with advanced loss functions and optimization strategies

## Installation

### Requirements
```
pandas
numpy
fastapi
wandb
torch
torch_geometric
scikit-learn
```

### Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Additional dependencies may need to be installed manually:
   ```
   pip install torch torch_geometric scikit-learn
   ```

## Usage

### Training
Run the main script to train the model:
```
python main.py
```

### Inference
Start the FastAPI server for real-time predictions:
```
uvicorn inference.model_server:app --reload
```

### Testing
Run the bandit integration demonstration:
```
python tests/demonstrate_bandit_integration.py
```

## Project Structure
```
CarbonIQ-2.0/
├── config/                 # Configuration files
├── data/                   # Data loading and graph building
├── inference/              # Model serving and API
├── models/                 # GNN and bandit models
├── training/               # Training pipeline
├── utils/                  # Helper functions
├── tests/                  # Test scripts
├── main.py                 # Main entry point
└── requirements.txt        # Project dependencies
```

## Model Configuration
The model can be configured through the `config/model_config.py` file, which includes parameters for:
- Model architecture (layers, dimensions)
- Training parameters (learning rate, batch size)
- Graph parameters (node limits, edge thresholds)
- Loss function weights

## Performance Monitoring
The project uses Weights & Biases (wandb) for experiment tracking and visualization.

## License
[Your License Information]

## Contributors
[Your Name/Team Information]
