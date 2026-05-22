## 'EOF' 


### 'EOF' 
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

```bash
cd ~/flowmatching
pip install -r requirements.txt
```

### .VolumeIcon.icns .file .nofollow .resolve .vol Applications Library System Users Volumes bin cores dev etc home opt private sbin tmp usr var 'EOF' 
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$

```bash
python test_imports.py
```

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }
```
 Bzier
 Losses
 Stage 1
 Stage 2
 Stage 3
 Metrics
```

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

```bash
cd experiments
python run_pipeline.py --config config.yaml
```

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }
```
============================================================
STAGE 0: Data preparation
============================================================
Generated toy data: (6000, 2000)
Conditions: 2 drugs  2 cell-lines  3 timepoints
...

============================================================
STAGE 1: Manifold extraction
============================================================
 State labels: [0 1 2]
 State distribution: [2000 2000 2000]
 Transition scores: min=0.XXX, max=0.XXX
...

============================================================
STAGE 2: CFM training
============================================================
 Model created: X.XXM parameters
  Epoch 5/50 | Loss: X.XXXX
  ...
  Epoch 50/50 | Loss: X.XXXX
 Training complete

============================================================
STAGE 3: Gene attribution
============================================================
 Gene decomposition:
  - Core drivers: X
  - Kinetic drivers: X
  - Static markers: X

============================================================
STAGE 4: Validation
============================================================
  mmd: X.XXXX
  wasserstein_2: X.XXXX

============================================================
 PIPELINE COMPLETE
============================================================
```

## .VolumeIcon.icns .file .nofollow .resolve .vol Applications Library System Users Volumes bin cores dev etc home opt private sbin tmp usr var 

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

```bash
# '
python -m data.synthetic_data_generator
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$6000 cells  2000 genes

# Bzier
python -m models.components.bezier
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } x(0)=x0, x(1)=x1

# 
python -m models.components.losses
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }L_flow, L_geom, L_context, L_smooth 

# Stage 1
python -m models.stage1_featuremap
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }3state clusters, transition scores

# Stage 2
python -m models.stage2_cfm
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

# Stage 3
python -m models.stage3_attribution
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }gene importance scores

# 
python -m validation.metrics
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }metrics computed
```

## 'EOF'

### Q: "No module named anndata"
A: `pip install anndata`

### Q: "No module named torch"
#            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_conda
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

### Q: Loss
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }
   -  learning_rate: 0.01
   -  lambda_smooth: 0.0001

### Q: 
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }
   -  batch_size:  8
   -  n_cells_per_condition: 250

### Q: GPU
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }
   -          device: cuda
#            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$ CUDA/cuDNN 
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } 

## 

1. EOF          
2. 'EOF'         `models/components/losses.py` L_context
3. 'EOF''EOF' synthetic_data_generator
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$Bzier vs 

---

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } README. QUICKSTART.md
