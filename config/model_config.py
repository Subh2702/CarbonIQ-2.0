class EnhancedGNNConfig:
    # Model Architecture - Improved parameters
    INPUT_DIM = 11              # Supplier features
    HIDDEN_DIM = 256            # Increased hidden layer size
    OUTPUT_DIM = 128            # Increased node embedding size
    NUM_LAYERS = 4              # More layers for better representation
    
    # Training Parameters - Optimized for better accuracy
    LEARNING_RATE = 0.0005      # Reduced learning rate for stability
    BATCH_SIZE = 16             # Smaller batch size for better gradients
    EPOCHS = 200                # More epochs with early stopping
    DROPOUT = 0.3               # Increased dropout for regularization
    
    # Graph Parameters
    MAX_NODES = 10000           # Maximum suppliers
    EDGE_THRESHOLD = 0.05       # Lower threshold for more edges
    
    # New parameters for enhanced training
    WEIGHT_DECAY = 1e-4         # L2 regularization
    GRADIENT_CLIP = 1.0         # Gradient clipping
    PATIENCE = 15               # Early stopping patience
    
    # Data augmentation
    NODE_FEATURE_NOISE = 0.05   # Add noise to node features
    EDGE_DROPOUT_RATE = 0.1     # Randomly drop edges during training
    
    # Loss weights
    FLOW_LOSS_WEIGHT = 0.4      # Carbon flow prediction
    CLASS_LOSS_WEIGHT = 0.3     # Supplier classification  
    LOCATION_LOSS_WEIGHT = 0.1  # Location prediction (auxiliary)
    PERFORMANCE_LOSS_WEIGHT = 0.1  # Performance prediction (auxiliary)
    CONTRASTIVE_LOSS_WEIGHT = 0.1  # Graph structure preservation