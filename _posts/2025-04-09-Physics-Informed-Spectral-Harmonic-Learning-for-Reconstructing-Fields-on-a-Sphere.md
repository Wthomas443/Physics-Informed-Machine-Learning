---
title: "Physics-Informed-Spectral-Harmonic-Learning-for-Reconstructing-Fields-on-a-Sphere"
date: "2025-04-09"
---
William Thomas – University of Nottingham

## Abstract

This dissertation explores the application of spectral harmonic methods—specifically spherical and vector spherical harmonics—for reconstructing noisy scalar and vector fields defined on the surface of a sphere. Motivated by the physical geometry of the Earth and the need for interpretable and data-efficient reconstructions of climate variables such as temperature and wind, the investigation proposes a physics-informed approach that bridges ideas from classical approximation theory and machine learning.

A least squares framework is used to fit spherical harmonic expansions to data, with additional regularization terms derived from Sobolev seminorms. These regularization strategies are designed to penalize non-smooth or physically implausible solutions, aligning the reconstructions with the governing physical principles of the underlying fields (e.g., diffusion, incompressibility). The methodology is validated on both synthetic data and real-world datasets including global climate measurements. For vector field reconstruction, divergence-free vector spherical harmonics are employed to ensure that physical constraints like mass conservation are strictly enforced.

The study finds that spherical harmonics provide a highly efficient basis for spatially coherent reconstruction on the sphere, and that careful regularization is crucial in balancing accuracy and overfitting. These insights form a foundation for future extensions into spatiotemporal modeling and adaptive spectral learning strategies.

---

## 1. Introduction

Recent advancements in computational power and data availability have accelerated the use of data-driven models for physical systems. However, many machine learning techniques lack a principled connection to the governing equations and structures of the domains they operate in. This dissertation seeks to close this gap by leveraging **spectral harmonic learning** on the sphere to perform high-fidelity reconstruction of scalar and vector fields from sparse and noisy observational data.

The motivation arises particularly from geophysical modeling: the Earth can be approximately treated as a sphere, and many climate and environmental fields naturally reside on this manifold. Traditional interpolation methods often struggle with irregular sampling and do not encode global physical constraints. By contrast, spherical harmonics are orthogonal basis functions that provide a global, multiscale representation well-suited to the geometry of the problem.

This study formulates the reconstruction problem as a regularized least squares optimization using spectral basis functions, applying it to problems in temperature field estimation and incompressible wind field recovery.

---

## 2. Spectral Harmonics on the Sphere

### 2.1 Spherical Harmonic Basis Functions

Spherical harmonics $ Y_\ell^m(\theta, \phi) $ are eigenfunctions of the Laplacian operator on the sphere $ S^2 $. They arise as solutions to the angular part of Laplace's equation when expressed in spherical coordinates. Formally, the basis functions satisfy:

$ \nabla^2_{\Omega} Y_\ell^m = -\ell(\ell+1) Y_\ell^m $

where $ \nabla^2_{\Omega} $ denotes the Laplacian on the sphere and $ (\theta, \phi) $ are the usual spherical angles.

Each $ Y_\ell^m $ is indexed by a degree $ \ell \geq 0 $ and an order $ -\ell \leq m \leq \ell $. They form a complete, orthonormal basis for the space $ L^2(S^2) $, allowing any square-integrable scalar function on the sphere to be expressed as a weighted sum.

### 2.2 Orthogonality and Inner Products

The spherical harmonics obey the orthogonality condition:

$ \langle Y_\ell^m, Y_{\ell'}^{m'} \rangle = \delta_{\ell \ell'} \delta_{m m'} $

which implies that their Fourier coefficients can be uniquely determined by projection onto the basis.

Gradients of spherical harmonics are also orthogonal:

$ \langle \nabla Y_\ell^m, \nabla Y_{\ell'}^{m'} \rangle = \ell(\ell+1) \delta_{\ell \ell'} \delta_{m m'} $

This allows higher-order smoothness penalties to be incorporated naturally in the spectral domain.

---

## 3. Scalar Field Reconstruction with Regularized Least Squares

### 3.1 Problem Setup

Given noisy observations $\{(\theta_i, \phi_i, f_i)\}_{i=1}^n$, we construct an approximation:

$S f(\theta, \phi) = \sum_{\ell=0}^L \sum_{m=-\ell}^\ell v_{\ell m} Y_\ell^m(\theta, \phi)$

and solve the regularized least squares problem:

$ \min_{v_{\ell m}} \sum_{i=1}^n |S f(\theta_i, \phi_i) - f_i|^2 + \lambda \sum_{\ell=0}^L \sum_{m=-\ell}^{\ell} (\ell(\ell+1))^k |v_{\ell m}|^2 $

Here, $ \lambda $ controls the strength of regularization and $ k $ determines the order of the Sobolev seminorm.

### 3.2 Matrix Formulation

The problem is linear in the coefficients. Let $ A \in \mathbb{C}^{n \times N} $ be the design matrix, where each row corresponds to the evaluation of the harmonics at a data point. Then the normal equations are:

$ v = (A^H A + \lambda M_k)^{-1} A^H f $

where $ M_k $ is a diagonal matrix with $ (\ell(\ell+1))^k $ along the diagonal.

---

## 4. Experiments on Synthetic Scalar Fields

Synthetic scalar fields are constructed from known combinations of spherical harmonics, allowing ground-truth comparison.

- **Experiment 1 (L=4, n=100):** The model successfully recovers the field with minimal error. Overfitting is observed for $ L > 4 $.
- **Experiment 2 (L=16, n=400):** Demonstrates the need for increased data to resolve higher-frequency components.

A heuristic relationship is empirically derived:

$ L \approx \frac{6}{7}\sqrt{n} $

Regularization using $ H^2 $ seminorms further reduces overfitting and leads to smoother reconstructions.

---

## 5. Application to Real Temperature Data

### 5.1 Small Dataset (n = 100)

Temperature measurements for 100 cities are reconstructed using $ L = 9 $. Cross-validation identifies optimal $ \lambda $ using test-error minimization. Despite sparse sampling, large-scale trends are captured.

### 5.2 Larger Dataset (n = 3510)

A higher-resolution dataset allows use of $ L = 12 $, improving reconstruction fidelity. The method struggles in unobserved regions (oceans, poles) due to lack of data. Regularization reduces artifacts and produces realistic reconstructions over land.

---

## 6. Vector Field Reconstruction with Incompressibility

### 6.1 Vector Spherical Harmonics

To reconstruct incompressible vector fields, a divergence-free basis is used:

$ \Phi_\ell^m(\theta, \phi) = \mathbf{r} \times \nabla Y_\ell^m(\theta, \phi) $

This guarantees that the approximation $ \mathbf{u}(\theta, \phi) = \sum v_{\ell m} \Phi_\ell^m $ satisfies $ \nabla \cdot \mathbf{u} = 0 $.

### 6.2 Implementation and Results

Design matrices for the $ \hat{\theta} $ and $ \hat{\phi} $ components are constructed, and the same least squares strategy is applied. Results on real wind data (n = 100, n = 500) demonstrate strong agreement in low-latitude regions, with challenges near singularities and under-sampled areas.

---

## 7. Conclusion and Outlook

This study shows that physics-informed spectral harmonic learning provides an effective and interpretable method for field reconstruction on the sphere. The use of Sobolev regularization ensures physically plausible approximations, and vector spherical harmonics enforce divergence-free constraints.

### Future Work

- Incorporating temporal dynamics (e.g., Fourier or Laguerre time bases)
- Adaptive selection of $ L $ and $ \lambda $
- Application to non-spherical manifolds
- Bayesian formulations for uncertainty quantification

This framework bridges the gap between classical approximation theory and modern data-driven modeling for geophysical applications.

