# Multi-Objective-LightGBM

Example

`bz_features_path=gs://etldata-prod-search-ranking-data-hkwv8r/data/shared/ranking/lightgbm/data/pyspark_buzzsaw_features_lambdamart_window_35days_perso_v2_newqtc/2020-11-28_2021-01-01/results/part-02546
`


`tree_config_path="gs://etldata-prod-search-ranking-data-hkwv8r/data/shared/ranking/lightgbm/tree_train_personalization.conf"`


`python3 lightgbm_ai_platform/run.py --bz-features-path $bz_features_path --tree-config-path $tree_config_path`
