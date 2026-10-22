# Data

The main classification experiments use a large feature table:

```text
data/new_features_3sec.csv
```

The original file is not included in this repository.

Its expected structure is:

```text
80,207 rows
59 columns total
57 numerical features
+ filename
+ length
```

If the original audio dataset is available, use notebook 01 to regenerate the feature table.

The `recovered_small/` directory contains smaller feature tables from earlier preprocessing experiments and is included only as sample data.
