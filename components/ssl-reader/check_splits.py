import sys, os
sys.path.insert(0, 'src')
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.disable(logging.CRITICAL)

from preprocessing_mediapipe import create_dataset_splits

splits, lm = create_dataset_splits(
    r'G:\research\Intelligent-Sinhala-Sign-Language-Communication-Platform\datasets\signVideo'
)

tr = len(splits['train'])
va = len(splits['val'])
te = len(splits['test'])
nc = len(lm)

print(f"Train: {tr}  ({100*tr/(tr+va+te):.0f}%)")
print(f"Val:   {va}  ({100*va/(tr+va+te):.0f}%)")
print(f"Test:  {te}  ({100*te/(tr+va+te):.0f}%)")
print(f"Total: {tr+va+te}")
print(f"Classes: {nc}")
print(f"Val per class (avg): {va/nc:.1f}")
print(f"Test per class (avg): {te/nc:.1f}")
