# TP Statistics – Regression & Machine Learning

ENSEIRB-MATMECA – ETE7MATH1 (Statistiques), P. Vallet.

## Goal

Study linear regression and its links with machine learning:

- **Part 1 – OLS, Ridge, LASSO** on synthetic data (`data1.npy`, `data2.npy`, `data3.npy`):
  - OLS fit and training error
  - Polynomial models (feature space)
  - Ridge and LASSO regularisation, effect of λ
- **Part 2 – Hyperspectral unmixing** on the real *Urban* image (307 × 307 pixels, 162 bands, 4 endmembers: asphalt, grass, tree, roof):
  - Abundance estimation with OLS and LASSO
  - Spectrum reconstruction and error
  - Abundance maps and simple segmentation
  - Bonus: constrained abundances (non-negative, sum to 1) with `scipy.optimize.nnls`

## Author

Dima Husseini
