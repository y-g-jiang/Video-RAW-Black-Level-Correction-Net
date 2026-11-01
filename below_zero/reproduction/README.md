# Cached score verification

From the `below_zero` directory:

```sh
python reproduction/reproduce_cached_tables.py --output cached_table_check.json
```

This verifies the six frozen methods on 138 sources and the common input identities for 552 cases. Neural inference is provided separately in `inference/`.
