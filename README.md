# Eigenvalue Comparison for Wreath Product Representations

This Python script computes and compares the smallest eigenvalues of Laplacians associated with different representations of the **wreath product** of a cyclic group \( C_m \) and a symmetric group \( S_n \). The script supports randomized edge weights, custom coefficient inputs, and batch testing over multiple group sizes.

---

## 🔧 Features

- Constructs the Laplacian matrix for the wreath product group element.
- Computes eigenvalues for:
  - Regular representation
  - Standard representation on \( C_m \times [n] \)
  - Standard representation on \( [n] \)
- Verifies theoretical inequalities between spectral gaps.
- Supports:
  - Randomized coefficients \( c_{ij} \)
  - Custom coefficient dictionary via JSON
  - Batch statistics mode to test spectral inequalities over 100 trials

---

## 📦 Requirements

- Python 3.x
- NumPy

You must also have the following Python files in the same directory:

- Sn_standard_representation_eigenvalues.py
- wreath_laplacian.py
- wreath_product.py
- wreath_regular_representation_eigenvalues.py
- wreath_standard_representation_eigenvalues.py

---

## 🚀 Usage

```python main.py -n 4 -m 3```

### Arguments

| Flag                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `-n`                      | Size of the action set for \( S_n \) (default: 3)                           |
| `-m`                      | Order of the cyclic group \( C_m \) (default: 2)                            |
| `-r`, `--randomize-c-ij`  | Randomize the coefficients \( c_{ij} \) from a normal distribution          |
| `-c`, `--c-dict`          | Path to a JSON file defining a dictionary with keys as tuple strings `(i, j)` and float values |
| `-s`, `--stats`           | Run 100 randomized trials for each combination of \( n \) and \( m \) and report timing and consistency |

---

## 📝 Example JSON input for `--c-dict`
```
{
  "(0, 1)": 1.0,
  "(0, 2)": 0.5,
  "(1, 2)": 0.8
}
```
---

## 📈 Output

The script prints:

- The smallest two eigenvalues for each representation
- Alerts if theoretical inequalities are violated or close to being violated
- Warnings if the input coefficient dictionary is malformed
- Timing statistics in `--stats` mode

---
