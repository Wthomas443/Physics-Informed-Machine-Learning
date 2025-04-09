---
title: Physics-Informed Spectral Harmonic Learning for Reconstructing Fields on the Sphere  
---
William Thomas – University of Nottingham

## Overview

This work presents a framework for reconstructing scalar and vector fields defined on a spherical domain using a spectral approach grounded in spherical harmonics and regularized least squares optimization. Motivated by challenges in geophysical modeling—where data is often sparse, noisy, and irregularly sampled—the proposed method combines the approximation-theoretic advantages of orthogonal basis expansions with regularization techniques drawn from variational methods and machine learning.

The methodology is validated on both synthetic and real-world datasets, including global temperature and wind data, and demonstrates robustness in recovering spatial structure under physical constraints such as smoothness and incompressibility.



## Spectral Approximation on the Sphere

Given a target function \( f : S^2 \to \mathbb{R} \), we consider a truncated spherical harmonic expansion of the form

\[
Sf(\theta, \phi) = \sum_{l=0}^L \sum_{m=-l}^l v_{lm} Y_l^m(\theta, \phi),
\]

where \( Y_l^m \) are the orthonormal spherical harmonics and \( v_{lm} \) are complex coefficients to be estimated. The spherical harmonics form a complete orthonormal basis for \( L^2(S^2) \), and any square-integrable function on the sphere admits such an expansion.

To approximate \( f \) from a finite set of noisy measurements \( \{f_i\}_{i=1}^n \) at locations \( \{(\theta_i, \phi_i)\} \), the coefficients \( \{v_{lm}\} \) are obtained by minimizing a regularized least squares functional:

\[
\min_{v_{lm}} \sum_{i=1}^n \left|Sf(\theta_i, \phi_i) - f_i \right|^2 + \lambda \mathcal{R}(Sf),
\]

where \( \mathcal{R} \) is a regularization term designed to enforce desirable properties in the solution.

---

## Regularization via Sobolev Norms

Two types of regularization are explored:

- **Tikhonov regularization (L² norm)**:
  \[
  \mathcal{R}(Sf) = \|Sf\|_{L^2(S^2)}^2 = \sum_{l=0}^L \sum_{m=-l}^l |v_{lm}|^2.
  \]

- **Sobolev seminorms (e.g., \( H^1 \), \( H^2 \))**:
  \[
  \mathcal{R}(Sf) = \sum_{l=0}^L \sum_{m=-l}^l (l(l+1))^k |v_{lm}|^2, \quad k \in \mathbb{N},
  \]

  corresponding to penalization of gradients (first derivatives) or Laplacians (second derivatives), consistent with smoothness or diffusive physical processes.

The regularization parameter \( \lambda \) is selected using techniques such as L-curve analysis or cross-validation on held-out data. In the case of physical systems governed by PDEs (e.g., the heat equation), Sobolev penalties offer a physically interpretable prior on the function space.

---

## Numerical Implementation and Analysis

The least squares system is formulated as \( A v = f \), where \( A \in \mathbb{C}^{n \times (L+1)^2} \) is the design matrix of spherical harmonics evaluated at the data points. The normal equations for the regularized problem take the form:

\[
v = (A^H A + \lambda M_k)^{-1} A^H f,
\]

where \( M_k \) is diagonal with entries \( (l(l+1))^k \), corresponding to the Sobolev seminorm used.

To improve numerical conditioning and avoid explicit squaring of the condition number (as in the normal equations), alternative solvers such as QR factorization are also employed.

The convergence of coefficients \( v_{lm} \) is shown to be \( \mathcal{O}(l^{-2k-1}) \) for \( f \in H^{2k}(S^2) \), leveraging the eigenvalue structure of the spherical Laplacian and classical results on series approximation.

---

## Synthetic and Real Data Results

**Synthetic Experiments:**  
Using functions constructed from known spherical harmonic combinations (up to degrees \( L=4 \) and \( L=16 \)), experiments validate the impact of regularization and model complexity. The empirical relationship between the number of data points \( n \) and the optimal harmonic degree \( L \) is found to scale approximately as:

\[
L \approx \frac{6}{7} \sqrt{n}, \quad N = (L + 1)^2,
\]

providing a guideline for model complexity selection under uniformly distributed data.

**Real Temperature Data (Berkeley Earth):**  
Temperature data from 100 and 3510 global cities were reconstructed for a fixed month (July 2003). Despite spatial sparsity and the absence of polar/ocean data, the model captured large-scale thermal trends. Optimal \( \lambda \) values were selected via k-fold cross-validation, and performance was evaluated on separate test sets.

---

## Vector Field Reconstruction with Incompressibility Constraint

To reconstruct tangential vector fields \( \mathbf{u} : S^2 \to \mathbb{R}^2 \), the approximation is extended using **vector spherical harmonics**:

\[
\Phi_l^m(\theta, \phi) = \mathbf{r} \times \nabla Y_l^m(\theta, \phi),
\]

which are divergence-free by construction: \( \nabla \cdot \Phi_l^m = 0 \). This ensures that the reconstructed field \( \mathbf{u} \) satisfies the incompressibility condition \( \nabla \cdot \mathbf{u} = 0 \), aligning with physical conservation laws in fluid dynamics and atmospheric science.

The least squares reconstruction of the vector field proceeds analogously, with separate design matrices for each component (in the \( \hat{\theta} \), \( \hat{\phi} \) directions), and regularization applied similarly via Sobolev-type seminorms.

---

## Conclusion

This dissertation demonstrates a principled framework for reconstructing scalar and vector fields on the sphere using physics-informed spectral learning. By leveraging the orthogonality and spectral properties of spherical harmonics and incorporating regularization strategies grounded in Sobolev space theory, the approach achieves robust reconstructions from sparse, noisy, and irregularly distributed data.

Applications to climate data validate the method's utility, particularly in contexts where adherence to physical constraints (e.g., smoothness, divergence-free behavior) is essential. Future extensions may incorporate temporal evolution (e.g., via Laguerre or Fourier time bases), anisotropic regularization, or data-driven basis adaptation for improved resolution and interpretability.

---

