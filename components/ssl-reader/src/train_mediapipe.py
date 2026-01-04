"""
Train sign language recognition model using MediaPipe features.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import json
from datetime import datetime

from preprocessing_mediapipe import MediaPipeFeatureExtractor, create_dataset_splits
from dataset import SinhalaSignLanguageDataset
from models import MultimodalLSTMModel, MultimodalTransformerModel, HybridModel

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logging.warning("TensorBoard not available. Install with: pip install tensorboard")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediaPipeTrainer:
    """Trainer for sign language models using MediaPipe features."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: Path,
        log_dir: Path = None
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        if HAS_TENSORBOARD and log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        learning_rate: float = 0.001,
        patience: int = 10
    ):
        """Full training loop with early stopping."""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint('checkpoint_best.pth', epoch, val_acc)
                logger.info(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Save latest model
            self._save_checkpoint('checkpoint_latest.pth', epoch, val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        if self.writer:
            self.writer.close()
    
    def _save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def test(self, test_loader):
        """Test the model."""
        # Load best checkpoint
        best_checkpoint_path = self.save_dir / 'checkpoint_best.pth'
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']+1}")
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = self.validate(test_loader, criterion)
        
        logger.info(f"Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # Save test results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'best_val_accuracy': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch)
        }
        
        with open(self.save_dir / 'test_results_mediapipe.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train sign language model with MediaPipe features')
    
    # Dataset arguments
    parser.add_argument('--dataset_root', type=str, 
                       default='datasets/signVideo_subset50',
                       help='Path to dataset root directory')
    
    # MediaPipe arguments
    parser.add_argument('--use_hands', action='store_true', default=True,
                       help='Use hand landmarks (default: True)')
    parser.add_argument('--use_pose', action='store_true', default=False,
                       help='Use pose landmarks (default: False - URL needs fixing)')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'hybrid'],
                       help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    # Directory arguments
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for TensorBoard logs')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory for cached features')
    
    # Preprocessing flag
    parser.add_argument('--preprocess', action='store_true',
                       help='Force preprocessing (ignore cache)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    workspace_root = project_root.parent.parent  # Go up to workspace root
    dataset_root = Path(args.dataset_root)
    
    if not dataset_root.is_absolute():
        dataset_root = workspace_root / dataset_root
    
    if args.cache_dir is None:
        cache_dir = project_root / 'data' / 'processed' / 'mediapipe'
    else:
        cache_dir = Path(args.cache_dir)
    
    if args.save_dir is None:
        save_dir = project_root / 'models' / 'mediapipe'
    else:
        save_dir = Path(args.save_dir)
    
    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = project_root / 'logs' / f'mediapipe_{timestamp}'
    else:
        log_dir = Path(args.log_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize MediaPipe feature extractor
    logger.info("Initializing MediaPipe feature extractor...")
    feature_extractor = MediaPipeFeatureExtractor(
        max_frames=args.max_frames,
        use_hands=args.use_hands,
        use_pose=args.use_pose
    )
    
    feature_dim = feature_extractor.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"  - Hands: {21*2*3 if args.use_hands else 0} dims (2 hands × 21 landmarks × 3 coords)")
    logger.info(f"  - Pose: {33*3 if args.use_pose else 0} dims (33 landmarks × 3 coords)")
    
    # Create dataset splits
    logger.info("Creating dataset splits...")
    splits, label_map = create_dataset_splits(
        str(dataset_root),
        max_frames=args.max_frames
    )
    
    num_classes = len(label_map)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(splits['train'])}")
    logger.info(f"Val samples: {len(splits['val'])}")
    logger.info(f"Test samples: {len(splits['test'])}")
    
    # Create datasets
    # Note: SinhalaSignLanguageDataset expects (samples, label_to_idx, feature_extractor, cache_dir, use_cache)
    # use_cache=False when preprocessing to force re-extraction
    use_cache = not args.preprocess
    
    train_dataset = SinhalaSignLanguageDataset(
        splits['train'], label_map, feature_extractor, cache_dir, 
        use_cache=use_cache
    )
    val_dataset = SinhalaSignLanguageDataset(
        splits['val'], label_map, feature_extractor, cache_dir,
        use_cache=use_cache
    )
    test_dataset = SinhalaSignLanguageDataset(
        splits['test'], label_map, feature_extractor, cache_dir,
        use_cache=use_cache
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    
    if args.model_type == 'lstm':
        model = MultimodalLSTMModel(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes
        )
    elif args.model_type == 'transformer':
        model = MultimodalTransformerModel(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            max_seq_len=args.max_frames
        )
    else:  # hybrid
        model = HybridModel(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_lstm_layers=args.num_layers,
            num_transformer_layers=args.num_layers,
            num_classes=num_classes
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Train
    trainer = MediaPipeTrainer(model, device, save_dir, log_dir)
    
    logger.info("Starting training...")
    trainer.train(
        train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience
    )
    
    # Test
    logger.info("Testing...")
    trainer.test(test_loader)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
