from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 12345
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler #a Sampler is an object used for running quantum circuits on a quantum device or simulator.
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# def plot_features(ax, features, labels, class_labels, marker, face, edge, label):
#     ax.scatter(
#         features[np.where(labels[:] == class_labels), 0],
#         features[np.where(labels[:] == class_labels), 1],
#         marker = marker,
#         facecolors = face,
#         edgecolors = edge,
#         label = label,
#     )

# def plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total):
#     plt.figure(figsize = (5,5))
#     plt.ylim(0, 2*np.pi)
#     plt.xlim(0, 2*np.pi)
#     plt.imshow(
#         np.asmatrix(adhoc_total).T,
#         interpolation = "nearest",
#         origin = "lower",
#         cmap = "RdBu",
#         extent = [0,2*np.pi, 0, 2*np.pi],
#     )

# adhoc_dimension = 2
# train_features, train_labels, test_features,test_labels, adhoc_total = ad_hoc_data(
#     training_size = 25,
#     test_size = 10,
#     n = adhoc_dimension,
#     gap = 0.6,
#     plot_data = False,
#     one_hot = False,
#     include_sample_total = True,
# )

def run_qiskit_qpca(data, n_components=2):
    data = np.array(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    feature_map = ZZFeatureMap(feature_dimension=data.shape[1], reps=2)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Compute the fidelity matrix
    fidelity_matrix = quantum_kernel.evaluate(data, data)
    
    # Perform eigen decomposition of the fidelity matrix
    eigenvalues, eigenvectors = np.linalg.eigh(fidelity_matrix)
    
    # Select the top components
    top_components = eigenvectors[:, -n_components:]
    reduced_data = np.dot(top_components.T, data ).T
    
    return reduced_data



def perform_quantum_pca(df, n_components: int = 2, method: str ='qiskit'):
    """
    Perform Quantum PCA using Qiskit on the provided DataFrame.

    Parameters:
    - df: DataFrame, the data on which to perform PCA
    - exclude_columns: List of columns to exclude from PCA
    - n_components: Integer, the number of principal components to reduce to
    - method: String, currently only 'qiskit' is supported

    Returns:
    - DataFrame containing the reduced components
    """
    # if exclude_columns:
    #     df = df.drop(columns=exclude_columns)

    data = df.values
    
    if method == 'qiskit':
        reduced_data = run_qiskit_qpca(data, n_components)
    else:
        raise ValueError("Currently only 'qiskit' method is supported")
    
    return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

#adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps = 2, entanglement = "linear")

# sampler = Sampler() # This allows you to use different backends (real quantum devices or simulators) as samplers.

# fidelity = ComputeUncompute(sampler = sampler)

# adhoc_kernel = FidelityQuantumKernel(fidelity = fidelity, feature_map = adhoc_feature_map)

# feature_map = ZZFeatureMap(feature_dimension = 2, reps = 2, entanglement = "linear")
# qpca_kernel = FidelityQuantumKernel(fidelity = fidelity, feature_map = feature_map)

# matrix_train = qpca_kernel.evaluate(x_vec = train_features)
# matrix_test = qpca_kernel.evaluate(x_vec = test_features, y_vec = train_features)

# from sklearn.decomposition import KernelPCA

# kernel_pca_rbf = KernelPCA(n_components = 2,kernel = "rbf")
# kernel_pca_rbf.fit(train_features)
# train_features_rbf = kernel_pca_rbf.transform(train_features)
# test_features_rbf = kernel_pca_rbf.transform(test_features)

# kernel_pca_q = KernelPCA(n_components = 2, kernel = "precomputed")
# train_features_q = kernel_pca_q.fit_transform(matrix_train)
# test_features_q = kernel_pca_q.transform(matrix_test)

# from sklearn.linear_model import LogisticRegression

# logistic_regression = LogisticRegression()
# logistic_regression.fit(train_features_q, train_labels)

# logistic_score = logistic_regression.score(test_features_q, test_labels)
# print(f"Logistic regression score: {logistic_score}")


# fig, (q_ax, rbf_ax) = plt.subplots(1,2,figsize = (10,5))

# plot_features(q_ax, train_features_q, train_labels, 0 , "s", "w", "b", "A train")
# plot_features(q_ax, train_features_q, train_labels, 1, "o", "w", "r", "B train")

# plot_features(q_ax, test_features_q, test_labels, 0, "s", "b", "w", "A test")
# plot_features(q_ax, test_features_q, test_labels, 1, "o", "r", "w", "A test")

# q_ax.set_ylabel("Principal component #1")
# q_ax.set_xlabel("Principal component #0")
# q_ax.set_title("Projection of training and test data\n using KPCA with Quantum Kernel")

# # Plotting the linear separation
# h = 0.01  # step size in the mesh

# # create a mesh to plot in
# x_min, x_max = train_features_q[:, 0].min() - 1, train_features_q[:, 0].max() + 1
# y_min, y_max = train_features_q[:, 1].min() - 1, train_features_q[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# predictions = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# predictions = predictions.reshape(xx.shape)
# q_ax.contourf(xx, yy, predictions, cmap=plt.cm.RdBu, alpha=0.2)

# plot_features(rbf_ax, train_features_rbf, train_labels, 0, "s", "w", "b", "A train")
# plot_features(rbf_ax, train_features_rbf, train_labels, 1, "o", "w", "r", "B train")
# plot_features(rbf_ax, test_features_rbf, test_labels, 0, "s", "b", "w", "A test")
# plot_features(rbf_ax, test_features_rbf, test_labels, 1, "o", "r", "w", "A test")

# rbf_ax.set_ylabel("Principal component #1")
# rbf_ax.set_xlabel("Principal component #0")
# rbf_ax.set_title("Projection of training data\n using KernelPCA")
# plt.show()