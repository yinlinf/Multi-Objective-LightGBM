#!/bin/bash

JOB_NAME="multi_lgbm_exp3_test15"
HPTUNING_CONFIG_FILE="./scripts/example_hp_config_big_disk.yaml"
PROJECT="etsy-sr-etl-prod"
STAGING_BUCKET=gs://etldata-prod-search-ranking-data-hkwv8r/
MACHINE_TYPE="n1-highcpu-32"
JOB_DIR="gs://etldata-prod-search-ranking-data-hkwv8r/user/yfu/moltr/"
PACKAGES=$(ls -p ./packages/* | tr '\n' ',')

gcloud ai-platform jobs submit training $JOB_NAME \
 --scale-tier=CUSTOM --master-machine-type=${MACHINE_TYPE} \
 --project=$PROJECT \
 --region=us-central1 \
 --staging-bucket=$STAGING_BUCKET \
 --job-dir=$JOB_DIR \
  --package-path=./lightgbm_ai_platform \
  --packages "$PACKAGES" \
  --module-name=lightgbm_ai_platform.run \
  --runtime-version=2.1 \
  --python-version=3.7 \
  --config $HPTUNING_CONFIG_FILE \
  --stream-logs






