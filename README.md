# Multi-Objective-LightGBM

Example

`bz_features_path=gs://etldata-prod-search-ranking-data-hkwv8r/data/shared/ranking/lightgbm/data/pyspark_buzzsaw_features_variant_b_lambdarank-window_21days/2020-11-19_2020-12-09/results/part-02500`


`tree_config_path="gs://etldata-prod-search-ranking-data-hkwv8r/data/shared/ranking/lightgbm/tree_train_lambdarank.conf"`


`python3 lightgbm_ai_platform/run.py --bz-features-path $bz_features_path --tree-config-path $tree_config_path`
