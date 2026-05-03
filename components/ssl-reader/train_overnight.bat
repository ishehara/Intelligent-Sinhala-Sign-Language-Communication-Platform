@echo off
echo ============================================
echo  Sinhala SSL - Overnight Training Run
echo  Started: %DATE% %TIME%
echo ============================================
echo.

set ROOT=g:\research\Intelligent-Sinhala-Sign-Language-Communication-Platform
set PYTHON=%ROOT%\venv\Scripts\python.exe
set SRC=%ROOT%\components\ssl-reader\src
set LOG=%ROOT%\components\ssl-reader\train_overnight.log

cd /d "%SRC%"

"%PYTHON%" train_mediapipe.py ^
    --model_type multistream ^
    --dataset_root datasets/signVideo ^
    --no_pose ^
    --use_hands ^
    --use_face ^
    --use_filtered_face ^
    --max_frames 60 ^
    --hidden_dim 256 ^
    --num_epochs 150 ^
    --batch_size 16 ^
    --learning_rate 0.001 ^
    --weight_decay 1e-4 ^
    --patience 30 ^
    --augment ^
    --device cuda ^
    2>&1 | tee "%LOG%"

echo.
echo ============================================
echo  Training finished: %DATE% %TIME%
echo  Log saved to: %LOG%
echo ============================================
pause
