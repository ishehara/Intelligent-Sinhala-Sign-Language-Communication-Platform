"""
Model optimization for on-device deployment.
Supports quantization, pruning, and export to ONNX/TFLite for edge devices.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize trained models for on-device deployment."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize model optimizer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for optimization
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.checkpoint = checkpoint
        self.label_to_idx = checkpoint.get('label_to_idx', {})
        
        # Load model architecture info
        from models import create_model
        from preprocessing import VideoFeatureExtractor
        
        extractor = VideoFeatureExtractor()
        input_dim = extractor.get_feature_dim()
        num_classes = len(self.label_to_idx)
        
        model_config = checkpoint.get('model_config', {
            'model_type': 'lstm',
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        })
        
        self.model = create_model(
            model_type=model_config.get('model_type', 'lstm'),
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            dropout=0.0  # Disable dropout for inference
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Original model size: {self._get_model_size_mb():.2f} MB")
    
    def _get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def quantize_dynamic(self, output_path: str = None) -> str:
        """
        Apply dynamic quantization (INT8) for faster inference on CPU.
        Reduces model size by ~4x with minimal accuracy loss.
        
        Args:
            output_path: Path to save quantized model
            
        Returns:
            Path to saved quantized model
        """
        logger.info("Applying dynamic quantization (INT8)...")
        
        # Dynamic quantization (works well for LSTM/Linear layers)
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.LSTM, nn.Linear},
            dtype=torch.qint8
        )
        
        # Calculate size reduction
        original_size = self._get_model_size_mb()
        
        if output_path is None:
            output_path = str(self.model_path.parent / f"{self.model_path.stem}_quantized.pth")
        
        # Save quantized model
        quantized_checkpoint = {
            'model_state_dict': quantized_model.state_dict(),
            'label_to_idx': self.label_to_idx,
            'model_config': self.checkpoint.get('model_config', {}),
            'optimization': 'dynamic_quantization_int8',
            'original_model': str(self.model_path)
        }
        
        torch.save(quantized_checkpoint, output_path)
        
        # Estimate new size (quantized model)
        saved_size = Path(output_path).stat().st_size / (1024 ** 2)
        
        logger.info(f"✓ Quantization complete!")
        logger.info(f"  Original size: {original_size:.2f} MB")
        logger.info(f"  Quantized size: {saved_size:.2f} MB")
        logger.info(f"  Reduction: {(1 - saved_size/original_size)*100:.1f}%")
        logger.info(f"  Saved to: {output_path}")
        
        return output_path
    
    def export_to_onnx(
        self,
        output_path: str = None,
        seq_length: int = 60,
        optimize: bool = True
    ) -> str:
        """
        Export model to ONNX format for cross-platform deployment.
        ONNX models can run on mobile, edge devices, and various frameworks.
        
        Args:
            output_path: Path to save ONNX model
            seq_length: Sequence length for export
            optimize: Whether to optimize ONNX model
            
        Returns:
            Path to saved ONNX model
        """
        logger.info("Exporting model to ONNX format...")
        
        if output_path is None:
            output_path = str(self.model_path.parent / f"{self.model_path.stem}.onnx")
        
        # Create dummy input
        from preprocessing import VideoFeatureExtractor
        extractor = VideoFeatureExtractor()
        input_dim = extractor.get_feature_dim()
        
        dummy_input = torch.randn(1, seq_length, input_dim).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Optimize ONNX model if requested
        if optimize:
            try:
                import onnx
                from onnxruntime.transformers import optimizer
                
                logger.info("Optimizing ONNX model...")
                onnx_model = onnx.load(output_path)
                
                # Basic optimization
                from onnx import shape_inference
                inferred_model = shape_inference.infer_shapes(onnx_model)
                onnx.save(inferred_model, output_path)
                
            except ImportError:
                logger.warning("onnx/onnxruntime not installed, skipping optimization")
        
        model_size = Path(output_path).stat().st_size / (1024 ** 2)
        
        logger.info(f"✓ ONNX export complete!")
        logger.info(f"  Model size: {model_size:.2f} MB")
        logger.info(f"  Saved to: {output_path}")
        logger.info(f"  Platform: Cross-platform (mobile, edge, web)")
        
        # Save metadata
        metadata = {
            'label_to_idx': self.label_to_idx,
            'input_shape': [1, seq_length, input_dim],
            'output_shape': [1, len(self.label_to_idx)],
            'model_type': self.checkpoint.get('model_config', {}).get('model_type', 'lstm')
        }
        
        metadata_path = str(Path(output_path).with_suffix('.json'))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Metadata saved to: {metadata_path}")
        
        return output_path
    
    def benchmark_inference(self, num_runs: int = 100) -> dict:
        """
        Benchmark inference speed for on-device performance.
        
        Args:
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        logger.info(f"Benchmarking inference speed ({num_runs} runs)...")
        
        from preprocessing import VideoFeatureExtractor
        extractor = VideoFeatureExtractor()
        input_dim = extractor.get_feature_dim()
        
        # Create dummy input
        dummy_input = torch.randn(1, 60, input_dim).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'device': self.device,
            'model_size_mb': self._get_model_size_mb(),
            'fps': 1000.0 / np.mean(times)
        }
        
        logger.info(f"✓ Benchmark complete!")
        logger.info(f"  Mean inference time: {results['mean_ms']:.2f} ms")
        logger.info(f"  Median: {results['median_ms']:.2f} ms")
        logger.info(f"  Min/Max: {results['min_ms']:.2f}/{results['max_ms']:.2f} ms")
        logger.info(f"  FPS: {results['fps']:.1f}")
        logger.info(f"  Device: {self.device}")
        
        # Check if it meets real-time requirements
        if results['mean_ms'] < 500:
            logger.info("  ✓ Meets real-time requirement (<500ms)")
        else:
            logger.warning("  ⚠ Exceeds real-time target (>500ms)")
        
        return results
    
    def create_edge_package(self, output_dir: str = None):
        """
        Create a complete package for edge deployment.
        Includes quantized model, ONNX model, and metadata.
        
        Args:
            output_dir: Directory to save edge package
        """
        if output_dir is None:
            output_dir = str(self.model_path.parent / 'edge_deployment')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating edge deployment package in {output_dir}")
        
        # 1. Quantize model
        quantized_path = self.quantize_dynamic(
            str(output_path / 'model_quantized.pth')
        )
        
        # 2. Export to ONNX
        onnx_path = self.export_to_onnx(
            str(output_path / 'model.onnx')
        )
        
        # 3. Benchmark
        benchmark_results = self.benchmark_inference()
        
        # 4. Save deployment info
        deployment_info = {
            'model_name': 'Sinhala Sign Language Recognizer',
            'version': '1.0.0',
            'num_classes': len(self.label_to_idx),
            'labels': self.label_to_idx,
            'models': {
                'quantized_pytorch': str(Path(quantized_path).name),
                'onnx': str(Path(onnx_path).name),
            },
            'benchmark': benchmark_results,
            'requirements': [
                'On-device processing (no internet required)',
                'Privacy-preserving (no data sent to cloud)',
                f'Real-time capable: {benchmark_results["fps"]:.1f} FPS',
                f'Model size: {benchmark_results["model_size_mb"]:.2f} MB'
            ],
            'deployment_platforms': [
                'Desktop (Windows/Linux/Mac)',
                'Mobile (iOS/Android via ONNX)',
                'Edge devices (Raspberry Pi, Jetson)',
                'Web (via ONNX.js)'
            ]
        }
        
        info_path = output_path / 'deployment_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ Edge deployment package created!")
        logger.info(f"{'='*60}")
        logger.info(f"Location: {output_path}")
        logger.info(f"\nContents:")
        logger.info(f"  - model_quantized.pth: Quantized PyTorch model")
        logger.info(f"  - model.onnx: ONNX model for cross-platform")
        logger.info(f"  - model.json: Label mapping and metadata")
        logger.info(f"  - deployment_info.json: Deployment guide")
        logger.info(f"\n{'='*60}")
        logger.info("ON-DEVICE DEPLOYMENT READY!")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize Sinhala Sign Language model for on-device deployment'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for optimized models')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply dynamic quantization')
    parser.add_argument('--export_onnx', action='store_true',
                       help='Export to ONNX format')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark inference speed')
    parser.add_argument('--edge_package', action='store_true',
                       help='Create complete edge deployment package')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for optimization (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelOptimizer(args.model_path, device=args.device)
    
    # If no specific operation, create edge package
    if not any([args.quantize, args.export_onnx, args.benchmark, args.edge_package]):
        args.edge_package = True
    
    # Perform requested operations
    if args.edge_package:
        optimizer.create_edge_package(args.output_dir)
    else:
        if args.quantize:
            optimizer.quantize_dynamic()
        
        if args.export_onnx:
            optimizer.export_to_onnx()
        
        if args.benchmark:
            optimizer.benchmark_inference()


if __name__ == "__main__":
    main()
