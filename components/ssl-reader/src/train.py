"""
Training script for Sinhala Sign Language Recognition model.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import create_data_loaders
from models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for sign language recognition model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        save_dir: str = '../models',
        log_dir: str = '../logs',
        patience: int = 10,
        label_to_idx: dict = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            log_dir: Directory for tensorboard logs
            patience: Early stopping patience
            label_to_idx: Label mapping dictionary
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.label_to_idx = label_to_idx
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / f'run_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Save directory: {self.save_dir}")
        logger.info(f"  Log directory: {self.log_dir}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]')
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]')
            
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def test(self) -> dict:
        """Test the model."""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'label_to_idx': self.label_to_idx
        }
        
        # Save latest checkpoint
        latest_path = self.save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print metrics
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}:")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Check for improvement
            is_best = False
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Test on best model
        logger.info("\nEvaluating on test set...")
        self.load_checkpoint(self.save_dir / 'checkpoint_best.pth')
        test_metrics = self.test()
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        
        # Save test results
        results = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'best_val_accuracy': self.best_val_acc
        }
        
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Sinhala Sign Language Recognition Model')
    
    # Data arguments
    parser.add_argument('--dataset_root', type=str, 
                       default='../../datasets/signVideo',
                       help='Root directory of the video dataset')
    parser.add_argument('--cache_dir', type=str,
                       default='../data/processed',
                       help='Directory to cache preprocessed features')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess all videos before training')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'hybrid'],
                       help='Type of model to use')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='../models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='../logs',
                       help='Directory for logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, label_to_idx, num_classes = create_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        cache_dir=args.cache_dir,
        preprocess=args.preprocess
    )
    
    # Get input dimension
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[-1]
    
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_frames
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        patience=args.patience,
        label_to_idx=label_to_idx
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
