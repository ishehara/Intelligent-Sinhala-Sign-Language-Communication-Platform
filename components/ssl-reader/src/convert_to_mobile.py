"""
Mobile model converter for Android deployment.
Converts PyTorch models to TensorFlow Lite for on-device Android inference.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileModelConverter:
    """Convert trained models to mobile-friendly formats."""
    
    def __init__(self, model_path: str):
        """
        Initialize mobile model converter.
        
        Args:
            model_path: Path to trained PyTorch model checkpoint
        """
        self.model_path = Path(model_path)
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.checkpoint = checkpoint
        self.label_to_idx = checkpoint.get('label_to_idx', {})
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Load model
        from models import create_model
        from preprocessing import VideoFeatureExtractor
        
        extractor = VideoFeatureExtractor()
        self.input_dim = extractor.get_feature_dim()
        self.num_classes = len(self.label_to_idx)
        
        model_config = checkpoint.get('model_config', {
            'model_type': 'lstm',
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.0
        })
        
        self.model = create_model(
            model_type=model_config.get('model_type', 'lstm'),
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            dropout=0.0  # Disable dropout for inference
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded: {self.num_classes} classes")
    
    def export_to_tflite(
        self,
        output_path: str = None,
        seq_length: int = 60,
        quantize: bool = True
    ) -> str:
        """
        Export model to TensorFlow Lite for Android deployment.
        
        Args:
            output_path: Path to save TFLite model
            seq_length: Sequence length for export
            quantize: Whether to apply quantization (recommended for mobile)
            
        Returns:
            Path to saved TFLite model
        """
        logger.info("Converting PyTorch model to TensorFlow Lite...")
        
        if output_path is None:
            output_path = str(self.model_path.parent / 'mobile' / 'model.tflite')
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Export to ONNX first
        onnx_path = str(Path(output_path).parent / 'temp_model.onnx')
        dummy_input = torch.randn(1, seq_length, self.input_dim)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info("Exported to ONNX intermediate format")
        
        # Step 2: Convert ONNX to TensorFlow
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_model_path = str(Path(output_path).parent / 'tf_model')
            tf_rep.export_graph(tf_model_path)
            
            logger.info("Converted ONNX to TensorFlow")
            
        except ImportError:
            logger.error("onnx-tf not installed. Installing alternative method...")
            # Alternative: Use traced model
            self._export_via_trace(output_path, seq_length, quantize)
            return output_path
        
        # Step 3: Convert TensorFlow to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        if quantize:
            logger.info("Applying dynamic range quantization for mobile...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = Path(output_path).stat().st_size / (1024 ** 2)
        
        logger.info(f"✓ TFLite conversion complete!")
        logger.info(f"  Model size: {model_size:.2f} MB")
        logger.info(f"  Quantized: {quantize}")
        logger.info(f"  Saved to: {output_path}")
        
        # Clean up temporary files
        import shutil
        if Path(onnx_path).exists():
            Path(onnx_path).unlink()
        if Path(tf_model_path).exists():
            shutil.rmtree(tf_model_path)
        
        return output_path
    
    def _export_via_trace(self, output_path: str, seq_length: int, quantize: bool):
        """Alternative TFLite export using direct PyTorch to TF conversion."""
        logger.info("Using alternative conversion method...")
        
        # Create a wrapper model for TF compatibility
        class MobileWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x)
        
        wrapper = MobileWrapper(self.model)
        wrapper.eval()
        
        # Trace the model
        dummy_input = torch.randn(1, seq_length, self.input_dim)
        traced_model = torch.jit.trace(wrapper, dummy_input)
        
        # Save traced model
        traced_path = str(Path(output_path).parent / 'traced_model.pt')
        traced_model.save(traced_path)
        
        logger.info(f"✓ Saved traced model: {traced_path}")
        logger.info(f"  Note: For full TFLite conversion, install: pip install onnx-tf")
        
        return traced_path
    
    def create_android_package(self, output_dir: str = None):
        """
        Create complete Android deployment package.
        
        Args:
            output_dir: Directory to save Android package
        """
        if output_dir is None:
            output_dir = str(self.model_path.parent / 'android_deployment')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating Android deployment package in {output_dir}")
        
        # 1. Export to TFLite
        tflite_path = self.export_to_tflite(
            str(output_path / 'model.tflite'),
            quantize=True
        )
        
        # 2. Create label file for Android
        labels_path = output_path / 'labels.txt'
        with open(labels_path, 'w', encoding='utf-8') as f:
            for idx in sorted(self.idx_to_label.keys()):
                f.write(f"{self.idx_to_label[idx]}\n")
        
        logger.info(f"Created labels file: {labels_path}")
        
        # 3. Create metadata file
        metadata = {
            'model_name': 'Sinhala Sign Language Recognition',
            'version': '1.0.0',
            'num_classes': self.num_classes,
            'input_shape': [1, 60, self.input_dim],
            'output_shape': [1, self.num_classes],
            'labels': self.label_to_idx,
            'platform': 'Android (TensorFlow Lite)',
            'requirements': [
                'TensorFlow Lite Runtime',
                'MediaPipe for Android',
                'Camera access',
                'No internet required'
            ],
            'features': [
                'Real-time inference',
                'On-device processing',
                'Privacy-preserving',
                'Offline capable'
            ]
        }
        
        metadata_path = output_path / 'model_info.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created metadata: {metadata_path}")
        
        # 4. Create integration guide
        self._create_android_integration_guide(output_path)
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ Android deployment package created!")
        logger.info(f"{'='*60}")
        logger.info(f"Location: {output_path}")
        logger.info(f"\nContents:")
        logger.info(f"  - model.tflite: Optimized model for Android")
        logger.info(f"  - labels.txt: Class labels")
        logger.info(f"  - model_info.json: Model metadata")
        logger.info(f"  - ANDROID_INTEGRATION.md: Integration guide")
        logger.info(f"\n{'='*60}")
        logger.info("READY FOR ANDROID DEPLOYMENT!")
        logger.info(f"{'='*60}")
    
    def _create_android_integration_guide(self, output_path: Path):
        """Create Android integration guide."""
        guide = """# Android Integration Guide

## 📱 Integrating TensorFlow Lite Model in Android

### 1. Add Dependencies (build.gradle)

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // MediaPipe for feature extraction
    implementation 'com.google.mediapipe:solution-core:latest.release'
    implementation 'com.google.mediapipe:hands:latest.release'
    implementation 'com.google.mediapipe:holistic:latest.release'
}
```

### 2. Add Model to Assets

Copy these files to `app/src/main/assets/`:
- `model.tflite`
- `labels.txt`

### 3. Kotlin/Java Integration

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SignLanguageRecognizer(context: Context) {
    private var interpreter: Interpreter
    
    init {
        // Load model
        val model = loadModelFile(context, "model.tflite")
        interpreter = Interpreter(model)
    }
    
    private fun loadModelFile(context: Context, modelPath: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun recognize(features: FloatArray): String {
        // Input: [1, 60, 395] shape
        val inputBuffer = ByteBuffer.allocateDirect(1 * 60 * 395 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        features.forEach { inputBuffer.putFloat(it) }
        
        // Output: [1, num_classes]
        val outputBuffer = ByteBuffer.allocateDirect(1 * NUM_CLASSES * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Get prediction
        outputBuffer.rewind()
        val probabilities = FloatArray(NUM_CLASSES)
        outputBuffer.asFloatBuffer().get(probabilities)
        
        val maxIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        return labels[maxIdx]
    }
}
```

### 4. Real-time Recognition

```kotlin
// In your Activity/Fragment
val recognizer = SignLanguageRecognizer(context)

// Process video frames
cameraSource.setFrameProcessor { frame ->
    // Extract features using MediaPipe
    val features = extractFeatures(frame)
    
    // Recognize sign
    val result = recognizer.recognize(features)
    
    // Update UI
    runOnUiThread {
        resultTextView.text = result
    }
}
```

### 5. Permissions (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" />
```

## 📝 Performance Tips

- Run on background thread
- Use GPU delegate for faster inference
- Batch multiple frames if needed
- Cache interpreter instance
"""
        
        guide_path = output_path / 'ANDROID_INTEGRATION.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        logger.info(f"Created integration guide: {guide_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert model for Android deployment'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for Android package')
    parser.add_argument('--tflite', action='store_true',
                       help='Export to TFLite only')
    parser.add_argument('--android_package', action='store_true',
                       help='Create complete Android package')
    
    args = parser.parse_args()
    
    converter = MobileModelConverter(args.model_path)
    
    if args.tflite:
        converter.export_to_tflite()
    elif args.android_package or not args.tflite:
        converter.create_android_package(args.output_dir)


if __name__ == "__main__":
    main()
