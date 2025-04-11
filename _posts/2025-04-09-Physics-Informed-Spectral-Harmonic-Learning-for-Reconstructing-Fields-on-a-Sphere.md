---
layout: posts
title: "Physics Informed Spectral Harmonic Learning for Reconstructing Fields on a Sphere"
date: "2025-04-09"
---
William Thomas – University of Nottingham

## Summary

This dissertation explores the application of spectral harmonic methods—specifically spherical and vector spherical harmonics—for reconstructing noisy scalar and vector fields defined on the surface of a sphere. Motivated by the physical geometry of the Earth and the need for interpretable and data-efficient reconstructions of climate variables such as temperature and wind, the investigation proposes a physics-informed approach that bridges ideas from classical approximation theory and machine learning.

A least squares framework is used to fit spherical harmonic expansions to data, with additional regularization terms derived from Sobolev seminorms. These regularization strategies are designed to penalize non-smooth or physically implausible solutions, aligning the reconstructions with the governing physical principles of the underlying fields (e.g., diffusion, incompressibility). The methodology is validated on both synthetic data and real-world datasets including global climate measurements. For vector fields, a divergence-free basis using vector spherical harmonics is constructed to model wind fields.

The study finds that spherical harmonics provide a highly efficient basis for spatially coherent reconstruction on the sphere, and that careful regularization is crucial in balancing accuracy and overfitting. These insights form a foundation for future extensions into spatiotemporal modeling and adaptive spectral learning strategies.

Recent advancements in computational power and data availability have accelerated the use of data-driven models for physical systems. However, many machine learning techniques lack a principled connection to the governing equations and structures of the domains they operate in. This dissertation seeks to close this gap by leveraging **spectral harmonic learning** on the sphere to perform high-fidelity reconstruction of scalar and vector fields from sparse and noisy observational data.

The motivation arises particularly from geophysical modeling: the Earth can be approximately treated as a sphere, and many climate and environmental fields naturally reside on this manifold. Traditional interpolation methods often struggle with irregular sampling and do not encode global physical constraints. By contrast, spherical harmonics are orthogonal basis functions that provide a global, multiscale representation well-suited to the geometry of the problem.

This study formulates the reconstruction problem as a regularized least squares optimization using spectral basis functions, applying it to problems in temperature field estimation and incompressible wind field recovery.

---

## 2. Spherical Harmonics

### 2.1 Spherical Harmonic Basis Functions
Spherical harmonics are defined as:

$$
Y^{\ell}_{m}:S^2\rightarrow \mathbb{C}, \hspace{1cm} Y^{\ell}_{m}(\vartheta,\varphi) = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}P_{\ell m}(\cos\vartheta)e^{im\varphi}.
$$

They exhibit spectral properties and form an orthonormal basis on a sphere. Any function on the sphere $$S^2 = \{(\vartheta,\varphi) \mid \vartheta \in [0, \pi],  \varphi \in [0, 2\pi)\}$$ can be written as an expansion of spherical harmonics:

$$
f(\vartheta, \varphi) = \sum^\infty_{\ell=0}\sum^\ell_{m=-\ell}v_{\ell m}Y^{\ell}_{m}(\vartheta,\varphi).
$$

They are eigenfunctions of the Laplacian operator on $$S^2$$. They arise as solutions to the angular part of Laplace's equation in spherical coordinates:

$$
\nabla^2_{\Omega} Y^{\ell}_{m} = -\ell(\ell+1) Y^{\ell}_{m}.
$$

Each function is indexed by degree $$\ell \geq 0$$ and order $$-\ell \leq m \leq \ell$$, and the set $$\{Y^{\ell}_{m}\}$$ forms the orthonormal basis.

### 2.2 Orthogonality and Inner Products

The harmonics satisfy:

$$
\langle Y^{\ell}_{m}, Y^{\ell'}_{m'} \rangle = \delta_{\ell \ell'} \delta_{m m'}
$$

and their gradients:

$$
\langle \nabla Y^{\ell}_{m}, \nabla Y^{\ell'}_{m'} \rangle = \ell(\ell+1) \delta_{\ell \ell'} \delta_{m m'}
$$

These properties allow for penalisation using different Sobolev seminorms of the solution.

---

## 3. Scalar Field Reconstruction

Given noisy observations $$\{(\vartheta_i, \varphi_i, f_i)\}_{i=1}^n$$, the scalar field is approximated as a truncated series expansion:

$$
Sf(\vartheta, \varphi) = \sum_{\ell=0}^L \sum_{m=-\ell}^{\ell} v_{\ell m} Y^{\ell}_{m}(\vartheta, \varphi).
$$

The minimisation problem to be solved is:

$$
\min_{v_{\ell m}} \sum_{i=1}^n |Sf(\vartheta_i, \varphi_i) - f_i|^2 + \lambda |Sf|_{H^k(S^2)}.
$$

The first term is the squared error which is to be minimised, the second is the regularisation term with regularisation parameter $$\lambda$$. It penalises with the seminorm of the kth Hilbert space, that is $||\nabla^k Sf||_{L^2(S^2)}$. So if it is required that the kth derivative of your solution exists or that $f \in H^k(S^2)$, this is a regularisation term that could be used to ensure this is the case. By use of the orthogonality of the spherical harmonics, it is true that:

$$
|Sf|_{H^k(S^2)} = \sum^L_{\ell=0}\sum^\ell_{m=-\ell}(\ell(\ell+1))^k|v_{\ell m}|^2.
$$

---

## 4. Numerical Implementation

Using matrix notation, let $$A$$ be the design matrix, and $$f$$ the vector of observations. Then the regularized solution is given by:

$$
v = (A^H A + \lambda M_k)^{-1} A^H f,
$$

where $$M_k$$ is a diagonal matrix with $$(\ell(\ell+1))^k$$ on the diagonal. The approach supports efficient solvers and cross-validation strategies for tuning $$\lambda$$.

---

## 5. Synthetic Field Experiments

### 5.1 Varying $$L$$ and $$n$$

For synthetic data constructed from known harmonics, reconstruction error decreases with increased sample size $$n$$. Empirically, the optimal truncation degree $$L$$ scales with:

$$
L \approx \frac{6}{7} \sqrt{n}.
$$

### 5.2 Regularization Effects

Without regularization, overfitting is observed as $$L$$ increases. Introducing Sobolev regularization (e.g., $$H^2$$) results in smoother reconstructions with better generalization.

---

## 6. Application to Temperature Data

### 6.1 Low-Resolution Dataset ($$n = 100$$)

Using $$L = 9$$ and $$H^2$$ regularization, the model captures broad-scale temperature gradients, but struggles in unsampled regions (e.g., oceans, poles).

### 6.2 High-Resolution Dataset ($$n = 3510$$)

Using $$L = 12$$ with the same framework yields improved detail and spatial resolution. Errors are reduced especially over land, demonstrating scalability.

---

## 7. Vector Field Reconstruction

To approximate divergence-free vector fields, we use vector spherical harmonics:

$$
\Phi^{\ell}_{m}(\vartheta, \varphi) = \mathbf{r} \times \nabla Y^{\ell}_{m}(\vartheta, \varphi).
$$

This guarantees $$\nabla \cdot \mathbf{u} = 0$$ by construction.

We expand:

$$
\mathbf{u}(\vartheta, \varphi) = \sum_{\ell=0}^L \sum_{m=-\ell}^{\ell} v_{\ell m} \Phi^{\ell}_{m}(\vartheta, \varphi),
$$

and fit coefficients using a similar regularized least squares procedure.

---

## 8. Real Wind Data Results

The method was applied to wind field observations at $$n = 100$$ and $$n = 500$$ sites. Reconstructions were accurate at low latitudes, with expected difficulties near the poles due to coordinate singularities and sparse data.

---

## 9. Conclusion

This work proposes a spectral learning framework based on spherical harmonics and vector harmonics for reconstructing fields on the sphere. Key contributions include:

- Use of Sobolev seminorm regularization in spectral fitting
- Empirical validation of $$L$$–$$n$$ scaling laws
- Application to real climate data
- Extension to incompressible vector fields

### Future Work

- Add time dependence (e.g., Fourier or Laguerre expansions)
- Explore non-spherical geometries
- Use Bayesian formulations for uncertainty quantification
- Investigate adaptive spectral refinement

---
