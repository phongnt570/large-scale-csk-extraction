# Refined Commonsense Knowledge from Large-Scale Web Contents

The pipeline is executed in the following order:

1. `nlp_pipeline.pipeline`
2. `open_ie.open_ie`
3. `triple_filtering.filter`
4. `triple_grouping.group_per_c4_part`,
`triple_grouping.group_all`,
`triple_grouping.get_frequent_triples`
5. `triple_clustering.precompute_embeddings`,
`triple_clustering.clustering`
6. `conceptnet_mapping.inference`
7. `ranking`
8. `final_filtering.final_filtering`

Global configurations can be found in `app_config.py`.
