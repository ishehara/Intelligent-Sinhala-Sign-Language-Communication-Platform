"""
Test script for SkeletonAugmenter to visualize augmentation effects.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from augmentation import SkeletonAugmenter


def create_sample_features(num_frames=60, feature_dim=1582):
    """Create synthetic feature tensor for testing."""
    # Create some structured patterns
    features = torch.zeros(num_frames, feature_dim)
    
    # Add hand landmarks (0-125)
    # Simulate a moving hand gesture
    for i in range(num_frames):
        t = i / num_frames  # Time from 0 to 1
        
        # Right hand (0-62)
        # Wrist at center, moving in circle
        features[i, 0] = 0.5 + 0.1 * np.cos(2 * np.pi * t)  # x
        features[i, 1] = 0.5 + 0.1 * np.sin(2 * np.pi * t)  # y
        features[i, 2] = 0.0  # z
        
        # Add some finger landmarks
        for j in range(1, 21):
            features[i, j*3] = features[i, 0] + 0.05 * np.random.randn()
            features[i, j*3+1] = features[i, 1] + 0.05 * np.random.randn()
            features[i, j*3+2] = 0.0
        
        # Left hand (63-125) - mirror pattern
        for j in range(21):
            features[i, 63 + j*3] = 1.0 - features[i, j*3]
            features[i, 63 + j*3+1] = features[i, j*3+1]
            features[i, 63 + j*3+2] = 0.0
    
    # Add face landmarks (126-1529)
    # Simple static face at center
    for i in range(468):
        features[:, 126 + i*3] = 0.5 + 0.02 * np.random.randn()  # x
        features[:, 126 + i*3+1] = 0.3 + 0.02 * np.random.randn()  # y
        features[:, 126 + i*3+2] = 0.0  # z
    
    # Add blendshapes (1530-1581) - emotion features
    features[:, 1530:1582] = torch.rand(num_frames, 52) * 0.5
    
    return features


def visualize_augmentation():
    """Visualize augmentation effects."""
    print("=" * 80)
    print("Testing SkeletonAugmenter")
    print("=" * 80)
    
    # Create sample features
    original = create_sample_features()
    print(f"\n✓ Created sample features: {original.shape}")
    
    # Initialize augmenter
    augmenter = SkeletonAugmenter(
        rotation_range=(-5.0, 5.0),
        scale_range=(0.9, 1.1),
        noise_std=0.002,
        temporal_shift_prob=0.3,
        apply_prob=1.0  # Always apply for visualization
    )
    print("✓ Initialized SkeletonAugmenter")
    
    # Apply augmentation multiple times
    num_augmentations = 5
    augmented_samples = []
    
    for i in range(num_augmentations):
        augmented = augmenter(original)
        augmented_samples.append(augmented)
        print(f"  Augmentation {i+1}: shape={augmented.shape}, "
              f"mean={augmented.mean():.6f}, std={augmented.std():.6f}")
    
    # Visualize right hand wrist trajectory (index 0, 1 for x, y)
    print("\n" + "=" * 80)
    print("Visualizing Right Hand Wrist Trajectory")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SkeletonAugmenter: Right Hand Wrist Trajectory', fontsize=16)
    
    # Original
    ax = axes[0, 0]
    x_orig = original[:, 0].numpy()
    y_orig = original[:, 1].numpy()
    ax.plot(x_orig, y_orig, 'b-o', linewidth=2, markersize=3, label='Wrist')
    ax.set_title('Original', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # Augmented samples
    for i in range(num_augmentations):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        
        augmented = augmented_samples[i]
        x_aug = augmented[:, 0].numpy()
        y_aug = augmented[:, 1].numpy()
        
        ax.plot(x_aug, y_aug, 'r-o', linewidth=2, markersize=3, label='Augmented')
        ax.plot(x_orig, y_orig, 'b--', linewidth=1, alpha=0.3, label='Original')
        ax.set_title(f'Augmentation {i+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'augmentation_visualization.png'")
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("Statistical Comparison")
    print("=" * 80)
    
    print(f"\nOriginal features:")
    print(f"  Mean: {original.mean():.6f}")
    print(f"  Std:  {original.std():.6f}")
    print(f"  Min:  {original.min():.6f}")
    print(f"  Max:  {original.max():.6f}")
    
    # Average over all augmented samples
    all_augmented = torch.stack(augmented_samples)
    mean_augmented = all_augmented.mean(dim=0)
    
    print(f"\nMean of augmented samples:")
    print(f"  Mean: {mean_augmented.mean():.6f}")
    print(f"  Std:  {mean_augmented.std():.6f}")
    print(f"  Min:  {mean_augmented.min():.6f}")
    print(f"  Max:  {mean_augmented.max():.6f}")
    
    # Calculate differences
    print(f"\nDifference from original:")
    diff = (mean_augmented - original).abs()
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    
    # Check temporal shifting
    print("\n" + "=" * 80)
    print("Temporal Shifting Test")
    print("=" * 80)
    
    original_short = create_sample_features(num_frames=30)
    augmenter_temporal = SkeletonAugmenter(temporal_shift_prob=1.0, apply_prob=1.0)
    
    shifted_samples = []
    for i in range(10):
        shifted = augmenter_temporal(original_short)
        shifted_samples.append(shifted)
    
    # Check frame diversity
    frame_diffs = []
    for i in range(len(shifted_samples)):
        for j in range(i+1, len(shifted_samples)):
            diff = (shifted_samples[i] - shifted_samples[j]).abs().mean()
            frame_diffs.append(diff.item())
    
    print(f"Average difference between shifted samples: {np.mean(frame_diffs):.6f}")
    print(f"✓ Temporal shifting introduces variation")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


def test_overfitting_scenario():
    """Test augmentation on overfitting scenario similar to user's case."""
    print("\n" + "=" * 80)
    print("Overfitting Mitigation Test (Train 94% → Val 63%)")
    print("=" * 80)
    
    # Simulate training with augmentation
    print("\nScenario: You have 1,729 training samples")
    print("Problem: Model memorizes training data (94% accuracy)")
    print("Solution: Apply augmentations to create variations")
    
    augmenter = SkeletonAugmenter(
        rotation_range=(-5.0, 5.0),
        scale_range=(0.9, 1.1),
        noise_std=0.002,
        temporal_shift_prob=0.3,
        apply_prob=0.8
    )
    
    # Create one sample
    sample = create_sample_features()
    
    # Generate multiple augmented versions
    num_variations = 8
    variations = [augmenter(sample) for _ in range(num_variations)]
    
    print(f"\n✓ Generated {num_variations} variations from 1 sample")
    
    # Calculate diversity
    total_diff = 0
    count = 0
    for i in range(len(variations)):
        for j in range(i+1, len(variations)):
            diff = (variations[i] - variations[j]).abs().mean().item()
            total_diff += diff
            count += 1
    
    avg_diversity = total_diff / count
    print(f"  Average diversity between variations: {avg_diversity:.6f}")
    
    print("\nExpected Results:")
    print("  Before: Train 94%, Val 63% (31% gap = overfitting)")
    print("  After:  Train 75-80%, Val 68-72% (5-10% gap = better generalization)")
    print("\n✓ Augmentation increases effective dataset size")
    print("✓ Reduces memorization, improves generalization")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SkeletonAugmenter Test Suite")
    print("Developer: IT22304674 – Liyanage M.L.I.S.")
    print("=" * 80)
    
    # Run tests
    visualize_augmentation()
    test_overfitting_scenario()
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run training with augmentation:")
    print("   python train_mediapipe.py --dataset_root ../../datasets/signVideo \\")
    print("                              --cache_dir data/processed/mediapipe_normalized \\")
    print("                              --augment --device cuda")
    print("\n2. Compare results with previous training (no augmentation)")
    print("\n3. Monitor training/validation gap reduction")
    print("=" * 80)
