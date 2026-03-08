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


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x if self.residual is None else self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    
    def __init__(self, input_dim, hidden_dims=[64, 128, 128], kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(hidden_dims)
        
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            dilation = 2 ** i
            
            layers.append(TemporalConvBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.network(x)     # (batch, output_dim, seq_len)
        
        # Global average pooling
        x = x.mean(dim=2)       # (batch, output_dim)
        
        return x


class AttentionFusion(nn.Module):
    """Attention-based fusion for multi-stream features."""
    
    def __init__(self, stream_dims, fusion_dim=512, dropout=0.3):
        super(AttentionFusion, self).__init__()
        
        self.num_streams = len(stream_dims)
        
        # Project each stream to fusion dimension
        self.stream_projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in stream_dims
        ])
        
        # Attention weights for each stream
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, stream_features):
        """
        Args:
            stream_features: List of tensors [(batch, dim1), (batch, dim2), ...]
        Returns:
            fused: (batch, fusion_dim)
            attention_weights: (batch, num_streams)
        """
        batch_size = stream_features[0].size(0)
        
        # Project all streams
        projected = []
        for i, features in enumerate(stream_features):
            proj = self.stream_projections[i](features)  # (batch, fusion_dim)
            projected.append(proj)
        
        # Stack streams
        stacked = torch.stack(projected, dim=1)  # (batch, num_streams, fusion_dim)
        
        # Compute attention weights for each stream
        attn_scores = self.attention(stacked).squeeze(-1)  # (batch, num_streams)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, num_streams)
        
        # Weighted combination
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (batch, num_streams, 1)
        fused = torch.sum(stacked * attn_weights_expanded, dim=1)  # (batch, fusion_dim)
        
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused, attn_weights


class MultiStreamFusionModel(nn.Module):
    """
    Multi-Stream Fusion Model for Sign Language Recognition.
    Separate streams for hands, face, and optional pose.
    
    Architecture:
        - Hand Stream: TCN (fast movements)
        - Face Stream: TCN + Attention (expressions + landmarks)
        - Pose Stream: Lightweight LSTM (optional)
        - Fusion: Attention-based weighted combination
    
    Developer: IT22304674 – Liyanage M.L.I.S.
    """
    
    def __init__(
        self,
        hand_dim: int = 126,
        face_dim: int = 180,  # Diagnostic: 60 key landmarks × 3, no blendshapes
        pose_dim: int = 0,    # Diagnostic: pose disabled
        num_classes: int = 227,
        hand_hidden: int = 128,
        face_hidden: int = 256,
        pose_hidden: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.3,  # Balanced dropout
        use_pose: bool = False  # Disabled for diagnostic run
    ):
        """
        Initialize Multi-Stream model.
        
        Args:
            hand_dim: Hand feature dimension (default: 126 for 2 hands)
            face_dim: Face feature dimension (default: 232 for filtered face + blendshapes)
            pose_dim: Pose feature dimension (default: 99 for body context)
            num_classes: Number of sign classes
            hand_hidden: Hand stream output dimension
            face_hidden: Face stream output dimension
            pose_hidden: Pose stream output dimension
            fusion_dim: Fusion layer dimension
            dropout: Dropout rate (AGGRESSIVE: 0.5 to combat data starvation)
            use_pose: Whether to use pose stream (now True by default)
        """
        super(MultiStreamFusionModel, self).__init__()
        
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        self.pose_dim = pose_dim
        self.use_pose = use_pose
        
        # Hand Stream: TCN for fast hand movements
        self.hand_stream = TemporalConvNet(
            input_dim=hand_dim,
            hidden_dims=[64, hand_hidden, hand_hidden],
            kernel_size=5,
            dropout=dropout
        )
        
        # Face Stream: TCN + Attention for facial expressions
        self.face_stream = TemporalConvNet(
            input_dim=face_dim,
            hidden_dims=[128, 256, face_hidden],
            kernel_size=5,
            dropout=dropout
        )
        
        # Pose Stream: Lightweight LSTM (optional)
        self.pose_stream = None
        if use_pose and pose_dim > 0:
            self.pose_stream = nn.LSTM(
                input_size=pose_dim,
                hidden_size=pose_hidden // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.pose_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Attention Fusion
        stream_dims = [hand_hidden, face_hidden]
        if use_pose and pose_dim > 0:
            stream_dims.append(pose_hidden)
        
        self.fusion = AttentionFusion(
            stream_dims=stream_dims,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Classifier with moderate dropout
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, total_features)
               Features are concatenated: [hand_features, face_features, pose_features]
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, num_streams) - stream importance
        """
        batch_size, seq_len, _ = x.shape
        
        # Split features into streams
        hand_features = x[:, :, :self.hand_dim]  # (batch, seq_len, hand_dim)
        face_features = x[:, :, self.hand_dim:self.hand_dim + self.face_dim]  # (batch, seq_len, face_dim)
        
        # Process each stream
        hand_out = self.hand_stream(hand_features)  # (batch, hand_hidden)
        face_out = self.face_stream(face_features)  # (batch, face_hidden)
        
        stream_features = [hand_out, face_out]
        
        # Process pose stream if enabled
        if self.use_pose and self.pose_dim > 0:
            pose_features = x[:, :, self.hand_dim + self.face_dim:]  # (batch, seq_len, pose_dim)
            pose_out, _ = self.pose_stream(pose_features)  # (batch, seq_len, pose_hidden)
            pose_out = pose_out.transpose(1, 2)  # (batch, pose_hidden, seq_len)
            pose_out = self.pose_pooling(pose_out).squeeze(-1)  # (batch, pose_hidden)
            stream_features.append(pose_out)
        
        # Fusion with attention
        fused, attention_weights = self.fusion(stream_features)  # (batch, fusion_dim), (batch, num_streams)
        
        # Classification
        logits = self.classifier(fused)  # (batch, num_classes)
        
        return logits, attention_weights


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('lstm', 'transformer', 'hybrid', or 'multistream')
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
    elif model_type == 'multistream':
        return MultiStreamFusionModel(num_classes=num_classes, **kwargs)
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
