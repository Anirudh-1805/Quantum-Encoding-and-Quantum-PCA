# Quantum Encoding and Quantum PCA

A Flask-based web application that provides **quantum data encoding** and **quantum principal component analysis (QPCA)** for classical datasets. Upload a CSV file, apply quantum-inspired transformations, and download the results — all through an intuitive web interface.

---

## ✨ Features

- **Amplitude Encoding** — Normalizes each row of numeric data into a unit-length quantum state vector (L2 normalization), preparing data for amplitude-based quantum algorithms.
- **Angle Encoding** — Maps classical feature values to qubit rotation angles via parameterized quantum gates:
  - **Rx (Pauli-X rotation)** — Encodes features as rotations around the X-axis.
  - **Ry (Pauli-Y rotation)** — Encodes features as rotations around the Y-axis.
  - **Rz (Pauli-Z rotation)** — Encodes features as rotations around the Z-axis.
- **Quantum PCA (QPCA)** — Performs dimensionality reduction using a quantum kernel built with Qiskit's `ZZFeatureMap` and `FidelityQuantumKernel`, followed by eigendecomposition for principal component extraction.
- **Batch Download** — When both encoding types are selected, results are bundled into a single `.zip` archive.

---

## 🏗️ Project Structure

```
Quantum-Encoding-and-Quantum-PCA/
├── app.py                  # Flask web server and route handlers
├── amplitude_encoding.py   # Amplitude encoding (L2 row normalization)
├── angle_encoding.py       # Angle encoding via Rx, Ry, Rz rotation gates
├── qpca.py                 # Quantum PCA using Qiskit quantum kernels
├── Iris.csv                # Sample dataset (Fisher's Iris dataset)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## 🔧 Prerequisites

- Python 3.8+
- pip

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Anirudh-1805/Quantum-Encoding-and-Quantum-PCA.git
   cd Quantum-Encoding-and-Quantum-PCA
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The Quantum PCA module additionally requires Qiskit packages. Install them with:
   > ```bash
   > pip install qiskit qiskit-algorithms qiskit-machine-learning
   > ```

---

## ▶️ Usage

### Running the Web App

```bash
python app.py
```

The Flask development server starts at `http://127.0.0.1:5000`. Open this URL in your browser.

### Workflow

1. **Upload** a CSV file (e.g., the included `Iris.csv`).
2. **Select** one or both encoding operations:
   - **Amplitude Encoding** — normalizes numeric columns row-wise.
   - **Angle Encoding** — choose a rotation gate (Rx, Ry, or Rz) to encode features.
3. **Download** the processed CSV file(s). If both encodings are selected, a `.zip` file containing both results is returned.

### Using Quantum PCA Programmatically

```python
import pandas as pd
from qpca import perform_quantum_pca

df = pd.read_csv("Iris.csv")

# Keep only numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Reduce to 2 principal components
reduced = perform_quantum_pca(numeric_df, n_components=2, method="qiskit")
print(reduced.head())
```

---

## 📖 Technical Details

### Amplitude Encoding

Each row of numeric features is divided by its L2 norm, producing a unit vector that can directly represent a quantum state's probability amplitudes:

$$|\psi\rangle = \frac{1}{\|x\|} \sum_{i} x_i |i\rangle$$

### Angle Encoding

Classical values are converted to rotation angles (in radians) and applied to single-qubit gates. For a feature value θ:

| Gate | Matrix |
|------|--------|
| **Rx(θ)** | `[[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]` |
| **Ry(θ)** | `[[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]` |
| **Rz(θ)** | `[[e^(-iθ/2), 0], [0, e^(iθ/2)]]` |

### Quantum PCA

1. Constructs a `ZZFeatureMap` quantum circuit with 2 repetitions.
2. Evaluates a **fidelity-based quantum kernel matrix** between all pairs of data points.
3. Performs eigendecomposition on the kernel matrix.
4. Projects data onto the top-*k* eigenvectors for dimensionality reduction.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| **Flask** | Web framework for the upload/download interface |
| **NumPy** | Numerical operations and linear algebra |
| **pandas** | Data manipulation and CSV I/O |
| **Qiskit** | Quantum circuit construction and simulation |
| **qiskit-algorithms** | Quantum algorithm utilities (state fidelity) |
| **qiskit-machine-learning** | Quantum kernel and ML integration |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anirudhan Ramkumar**
