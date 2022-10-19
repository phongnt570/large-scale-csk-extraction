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

Files needed for the pipeline to run are:

- Precomputed similarity scores between C4 documents and Wikipedia articles:
  https://nextcloud.mpi-inf.mpg.de/index.php/s/nJSSW5QBQR3XoxH
  (cf. [triple_filtering/filter.py](`triple_filtering/filter.py`))
- Subjects: https://nextcloud.mpi-inf.mpg.de/index.php/s/TiSm3rrJ9kEqfm8
  (cf. [triple_filtering/filter.py](`triple_filtering/filter.py`))
- ConceptNet mapping train/dev
  files: https://nextcloud.mpi-inf.mpg.de/index.php/s/JeLRgsiNykcnRbs
  (cf. [conceptnet_mapping/finetune.py](`conceptnet_mapping/finetune.py`))

If you use Ascent++, please cite the following paper:

```bibtex
@ARTICLE{ascentpp,
  author={Nguyen, Tuan-Phong and Razniewski, Simon and Romero, Julien and Weikum, Gerhard},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Refined Commonsense Knowledge from Large-Scale Web Contents}, 
  year={2022},
  doi={10.1109/TKDE.2022.3206505}
}
```
