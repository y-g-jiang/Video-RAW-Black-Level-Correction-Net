# Below Zero: Spatially Adaptive Conditional Readout Sampling for Lower-Censored RAW Images

Yaogeng Jiang · Independent Researcher · y.g.jiang@sjtu.edu.cn

[Paper](paper/en/main.pdf) · [Supplement](paper/en/supplement.pdf) · [Visual A](paper/en/visual_en.pdf) · [中文](paper/zh/main.pdf)

## Inference

```sh
python -m pip install -r inference/requirements.txt
python inference/make_demo.py --output demo_input.npz
python inference/run_inference.py --input demo_input.npz --output signed_output.npz --device cpu
```

Use `--device cuda` for a compatible GPU. Input `Y` is a nonnegative integer array `[R,G1,G2,B]` in native DN with packed height and width divisible by 16. `b` contains four residual offsets in DN, within the frozen training range ±0.25 DN. Optional `valid` has shape `[1,H,W]`. Pad to a multiple of 16 with zero observations and zero validity, and crop after inference. Output contains `signed`, `signal_DN`, and `zero_probability`; the probability map is populated only at valid zero candidates. Positive codes are unchanged and reclipping exactly reproduces `Y`.

The released model is seed 7103, step 80000, with SHA-256 `662eef3e581eb77d1e948ee800d6e63c0ab0039c3b4cff8f9fafe8a99074894f`. Gain is 21.601092412099675 DN/e, analog read noise is 24.95960511926749 DN, and the tail boundary is K=304. These are the frozen ISO12800 settings.

## Numerical reproduction

```sh
python reproduction/reproduce_cached_tables.py --output cached_table_check.json
python reproduction/physics/smoke.py --output physics_check
```

`reproduction/independent_test/` contains the frozen six-method case scores and bootstrap summary. `results/paper_data/` contains sourcewise, regional, real-readout and stress-test scores. `evidence/clean_color/` contains block, render and source scores. `configs/` and `protocol/` preserve the registered parameters; `datasets/` contains identifiers and preparation code, not source photographs. Archived metadata may retain original absolute paths as provenance. The inference and cached-score commands above are portable entry points.

## Paper sources

The editable sources are in `paper/source/`. With XeLaTeX, BibTeX, IEEEtran, TeX Gyre fonts and the required packages installed, run XeLaTeX on `main.tex`, BibTeX on `main`, then XeLaTeX on `supplement.tex` and `main.tex` twice in `paper_en/`. Compile the Chinese version with Fandol fonts. Keep each main, supplement and visual PDF together to retain cross-file links.

## License

Original code and the trained weights: [MIT](LICENSE-CODE). Original manuscript, figures and numerical results: [CC BY 4.0](LICENSE-CONTENT). Third-party materials and source datasets retain their own terms; see [notices](THIRD_PARTY_NOTICES.md). `MANIFEST.json` records file SHA-256 values.
