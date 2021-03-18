#!/bin/bash

JOB_NAME="multi_lgbm_freshness_5"
HPTUNING_CONFIG_FILE="./scripts/example_hp_config_big_disk_7days.yaml"
PROJECT="etsy-sr-etl-prod"
STAGING_BUCKET=gs://etldata-prod-search-ranking-data-hkwv8r/
MACHINE_TYPE="n1-highmem-96"
PACKAGES=$(ls -p ./packages/* | tr '\n' ',')
JOB_DIR=gs://etldata-prod-search-ranking-data-hkwv8r/ranking/lightgbm_experiments

gcloud ai-platform jobs submit training $JOB_NAME \
  --scale-tier=CUSTOM --master-machine-type=${MACHINE_TYPE} \
  --project=$PROJECT \
  --region=us-central1 \
  --job-dir=$JOB_DIR \
  --package-path=./lightgbm_ai_platform \
  --packages "$PACKAGES" \
  --module-name=lightgbm_ai_platform.run \
  --runtime-version=2.1 \
  --python-version=3.7 \
  --config $HPTUNING_CONFIG_FILE \
  --stream-logs






