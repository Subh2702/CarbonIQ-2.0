class GNNConfig:
    # Model Architecture
    INPUT_DIM = 15              # Supplier features
    HIDDEN_DIM = 128            # Hidden layer size
    OUTPUT_DIM = 64             # Node embedding size
    NUM_LAYERS = 3              # GNN layers
    
    # Training Parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    DROPOUT = 0.2
    
    # Graph Parameters
    MAX_NODES = 10000           # Maximum suppliers
    EDGE_THRESHOLD = 0.1        # Minimum carbon flow for edge
    
class BanditConfig:
    NUM_ARMS = 50               # Top suppliers to consider
    EXPLORATION_RATE = 0.1      # Epsilon for epsilon-greedy
    REWARD_WINDOW = 100         # Recent performance window
    
class TransformerConfig:
    SEQUENCE_LENGTH = 30        # Days of historical data
    D_MODEL = 256               # Transformer dimension
    N_HEADS = 8                 # Attention heads
    N_LAYERS = 6                # Transformer layers