"""
Neural network models for Sinhala Sign Language Recognition.
Implements multimodal LSTM/Transformer architectures.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultimodalLSTMModel(nn.Module):
    """
    Multimodal LSTM model for sign language recognition.
    Processes temporal sequences of hand, face, and pose features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize the multimodal LSTM model.
        
        Args:
            input_dim: Input feature dimension per frame
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of sign classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(MultimodalLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension after LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention layer to focus on important frames
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_dim)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_output_dim)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits


class MultimodalTransformerModel(nn.Module):
    """
    Multimodal Transformer model for sign language recognition.
    Uses self-attention for temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 100,
        dropout: float = 0.3,
        max_seq_len: int = 60
    ):
        """
        Initialize the multimodal Transformer model.
        
        Args:
            input_dim: Input feature dimension per frame
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of sign classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(MultimodalTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridModel(nn.Module):
    """
    Hybrid model combining LSTM and Transformer.
    Uses LSTM for low-level temporal features and Transformer for high-level reasoning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        max_seq_len: int = 60
    ):
        """
        Initialize the hybrid model.
        
        Args:
            input_dim: Input feature dimension per frame
            hidden_dim: Hidden dimension of LSTM
            d_model: Dimension of the Transformer model
            nhead: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            num_transformer_layers: Number of Transformer layers
            num_classes: Number of sign classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(HybridModel, self).__init__()
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Project LSTM output to d_model
        self.projection = nn.Linear(lstm_output_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer for high-level reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_dim)
        
        # Project to d_model
        x = self.projection(lstm_out)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('lstm', 'transformer', or 'hybrid')
        input_dim: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional arguments for the model
        
    Returns:
        Initialized model
    """
    if model_type == 'lstm':
        return MultimodalLSTMModel(input_dim, num_classes=num_classes, **kwargs)
    elif model_type == 'transformer':
        return MultimodalTransformerModel(input_dim, num_classes=num_classes, **kwargs)
    elif model_type == 'hybrid':
        return HybridModel(input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    batch_size = 4
    seq_len = 60
    input_dim = 395  # From preprocessing
    num_classes = 100
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing LSTM Model:")
    lstm_model = create_model('lstm', input_dim, num_classes)
    lstm_out = lstm_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {lstm_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    print("\nTesting Transformer Model:")
    transformer_model = create_model('transformer', input_dim, num_classes)
    transformer_out = transformer_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {transformer_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    print("\nTesting Hybrid Model:")
    hybrid_model = create_model('hybrid', input_dim, num_classes)
    hybrid_out = hybrid_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {hybrid_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
