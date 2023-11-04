#!/bin/bash

NOTE=$1

# Check if WANDB_MODE is provided
if [ -z "$2" ]; then
  echo "Please enter the mode for wandb (default is 'run'): "
  read WANDB_MODE
  if [ -z "$WANDB_MODE" ]; then
    WANDB_MODE="run"
  fi
else
  WANDB_MODE=$2
fi

# Add an underscore before NOTE if it's set
if [ -n "$NOTE" ]; then
  NOTE="_${NOTE}"
fi

# Check if DATASET_CHOICE is provided
if [ -z "$3" ]; then
  echo "Please enter the dataset choice (CN, US, EU or default 'all'): "
  read DATASET_CHOICE
  if [ -z "$DATASET_CHOICE" ]; then
    DATASET_CHOICE="all"
  fi
else
  DATASET_CHOICE=$3
fi

# Check if SEED_CHOICE is provided
if [ -z "$4" ]; then
  echo "Please enter the seed choice (default is 'all'): "
  read SEED_CHOICE
  if [ -z "$SEED_CHOICE" ]; then
    SEED_CHOICE="all"
  fi
else
  SEED_CHOICE=$4
fi

# Check if LAG_HORIZON_CHOICE is provided
echo "Please enter the lag-horizon choice:"
echo "1: lag=7, horizon=1 (default)"
echo "2: lag=7, horizon=3"
echo "3: both"
read LAG_HORIZON_CHOICE
case $LAG_HORIZON_CHOICE in
  1)
    LAG_HORIZONS=("7_1")
    ;;
  2)
    LAG_HORIZONS=("7_3")
    ;;
  *)
    LAG_HORIZONS=("7_1" "7_3")
    ;;
esac

# Define seeds and datasets
SEEDS=(42 56 89 159 200)
DATASETS=("CN" "US" "EU")

# Function to run the model with given parameters
run_model() {
  local SEED=$1
  local DATASET=$2
  local LAG_HORIZON=$3
  local LAG=${LAG_HORIZON%%_*}
  local HORIZON=${LAG_HORIZON##*_}
  
  # Define your directories and run name
  DIR="outputs/AGCRN_time_dependent_matrix_laplacian/\${now:%m-%d_%H-%M-%S}${NOTE}_dataset${DATASET}_lag${LAG}_horizon${HORIZON}_seed${SEED}"
  RUN_NAME="\${now:%m-%d_%H-%M-%S}${NOTE}_dataset${DATASET}_lag${LAG}_horizon${HORIZON}_seed${SEED}"

  # Run your Python script with the necessary parameters
  python src/models/AGCRN_time_dependent_matrix_laplacian/Run.py \
    "hydra.run.dir=${DIR}" \
    "log_dir=${DIR}" \
    "wandb_dir=${DIR}" \
    "run_name=${RUN_NAME}" \
    "wandb_mode=${WANDB_MODE}" \
    "seed=${SEED}" \
    "dataset=${DATASET}" \
    "lag=${LAG}" \
    "horizon=${HORIZON}"
}

# Iterate through datasets
for DATASET in "${DATASETS[@]}"; do
  if [ "$DATASET_CHOICE" = "all" ] || [ "$DATASET_CHOICE" = "$DATASET" ]; then
    
    # Iterate through Lag-Horizon combinations
    for LAG_HORIZON in "${LAG_HORIZONS[@]}"; do

      # If SEED_CHOICE is "all", then iterate through all seeds
      if [ "$SEED_CHOICE" = "all" ]; then
        for SEED in "${SEEDS[@]}"; do
          run_model "$SEED" "$DATASET" "$LAG_HORIZON"
        done
      else
        # Else, just use the provided seed
        run_model "$SEED_CHOICE" "$DATASET" "$LAG_HORIZON"
      fi
      
    done
  fi
done
