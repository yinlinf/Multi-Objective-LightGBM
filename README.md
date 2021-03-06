# Multi-Objective-LightGBM

## Overview
This repository provides the option of training LightGBM models with customized objective function.
Note the repo is forked from this https://github.etsycorp.com/ebenjamin/LightGBM_HP_Tuning repo for hyperparameter tuning. 
The usages are similar. 

## Usage
In all commands below, run the command from the repository root.

## To run:

1. Run bash ./scripts/create_packages.sh. This will create .tar.gz package files for Dresden and Buzzsaw which AI Platform needs to import those libraries. It will save these to ./packages.
You only need to do this once!
 
2. Create a HP tuning YAML file:
 * See example_hp_config.yaml for an example on small data, and example_hp_config_big_disk.yaml for an example on full production data making use of more disk space.
 
 * The only mandatory args are:
   * --bz-features-path: Path to buzzsaw feature files for model training/validation.
   * --bz_features_path_test: Path to buzzsaw features files for model testing. Typically use next day's data to test.
   * --tree-config-path: Path to a LightGBM tree configuration file, e.g gs://etldata-prod-search-ranking-data-hkwv8r/data/shared/ranking/lightgbm/tree_train_lambdarank.conf

3. Modify and run ./scripts/hp_tune_ai_platform.sh to your liking:
You may want to edit JOB_NAME, HPTUNING_CONFIG_FILE, PROJECT, and STAGING_BUCKET
4. You now monitor your jobs on the AI Platform Jobs Page! When you click on your job you can then see validation score for each trial! Image of Job
