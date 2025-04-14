---
layout: posts
title: "Physics Informed Spectral Harmonic Learning for Reconstructing Fields on a Sphere"
date: "2025-04-09"
---
William Thomas – University of Nottingham

## 1. Summary

This post summarises the work I completed, in my fourth year at the University of Nottingham, for my Dissertation module. It explores the application of physics informed machine learning methods, specifically spherical and vector spherical harmonics in a truncated series expansion for reconstructing noisy scalar and vector fields which are defined on the surface of a sphere. Motivated by the physical geometry of the Earth and the need for interpretable and efficient reconstructions of climate variables such as temperature and wind, the investigation creates a simple model to make spatial reconstructions of such data on a sphere.

A least squares framework is used to fit unknown coefficients to data, with additional regularisation terms  explored. These regularisation methods are designed to penalize non smooth or physically implausible solutions, aligning the reconstructions with the governing physical principles of the underlying fields (e.g., diffusion, incompressibility). The method is tested on both synthetic data and real world datasets including global climate measurements. For vector fields, a divergence-free basis using vector spherical harmonics is constructed to model wind fields.

The study finds that spherical harmonics provide a highly efficient basis for spatial reconstruction on the sphere, and that careful regularisation is crucial in balancing accuracy and overfitting. These insights form a foundation for future extensions into spatiotemporal modelling.

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

The first term is the squared error which is to be minimised, the second is the regularisation term with regularisation parameter $$\lambda$$. It penalises with the seminorm of the kth Hilbert space, that is $$\| \nabla ^k Sf\|_{L^2(S^2)}$$. So if it is required that the kth derivative of your solution exists or that $$f \in H^k(S^2)$$, this regularisation term can be used to ensure it is the case. By use of the orthogonality of the spherical harmonics, it is true that:

$$
|Sf|_{H^k(S^2)} = \sum^L_{\ell=0}\sum^\ell_{m=-\ell}(\ell(\ell+1))^k|v_{\ell m}|^2.
$$

The minimisation problem is now 

$$
\min_{v_{\ell m}} \sum_{i=1}^n |Sf(\vartheta_i, \varphi_i) - f_i|^2 + \lambda \sum^L_{\ell=0}\sum^\ell_{m=-\ell}(\ell(\ell+1))^k|v_{\ell m}|^2.
$$

### 3.1 Numerical Implementation

Using matrix notation, let $$A$$ be the design matrix, and $$f$$ the vector of observations. Then the regularized solution is given by:

$$
v = (A^H A + \lambda M_k)^{-1} A^H f,
$$

where $$M_k$$ is a diagonal matrix with $$(\ell(\ell+1))^k$$ on the diagonal.

---

## 4. Model Analysis with Synthetic Data
The model can be tested using synthetic data. This data is created with a truncated series of spherical harmonics then noise is added from some normal distribution of values with mean $$0$$. It is found that to optimally approximate data created with a series truncated with value $$L$$, you need an approximation made with the same value $$L$$.

### 4.1 Varying $$L$$ and $$n$$

For synthetic data constructed from a known series of harmonics, reconstruction error decreases with increased sample size $$n$$. The optimal value of $$L$$, the maximum degree of spherical harmonic used, has the following apprximate relationship:

$$
L \approx \frac{6}{7} \sqrt{n}.
$$

![$$L$$ and $$n$$ relation]({{"/assets/images/n-vs-L.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 1:The relationship between the optimal maximum degree of spherical harmonic $$L$$ and the number of data points $$n$$ for unregularised models*
{:.caption}

The blue line in Figure 1 shows the true relaationship, found from the experimental synthetic data.

### 4.2 Regularisation Effects

Without regularisation, overfitting is observed as $$L$$ increases. Introducing regularisation with Sobolev seminorms (e.g., $$H^2$$) results in smoother reconstructions without the effects of overfitting.

![Max degrees]({{"/assets/images/Complexity with sobolev reg.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 2: Errors from approximations made with a regularised model and different values of $$L$$*
{:.caption}

Optimal values for regularisation parameter value $$\lambda$$ can be found using L-curves: a trade-off graph ballencing the size of  the coefficients' norm and the residual norm. The optimal value of $$\lambda$$ being found at the "corner" of the L.

![L-curve]({{"/assets/images/L-curve for sobolev penalisation.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 3: L-curve for an approximation of $$L=16$$.*
{:.caption}

---

## 5. Application to Temperature Data

### 5.1 Low-Resolution Dataset ($$n = 100$$)

Using $$L = 9$$ and $$H^2$$ regularisation, the model captures broad scale temperature patterns, but struggles in unsampled regions (e.g., oceans, poles). 

![Major city data]({{"/assets/images/Major cities data.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 4: Temperature mesurements in major cities on the globe over the month 07/2003 in degrees Celsisus*
{:.caption}

Errors over different folds of test data is plotted to find an optimal $\lamda$ value. This is because plotting the trade-off between the residual norm and the coefficient norm doesn't produce an $$L$$ shape for real data.

![Major city lamda]({{"/assets/images/Test errors major cities.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 5: Test errors for different folds of data*
{:.caption}

![Major city approx]({{"/assets/images/Major cities approx.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 6: Resconstruction of the data in Figure 4 in degrees Celsisus*
{:.caption}

### 5.2 High-Resolution Dataset ($$n = 3510$$)

Using $$L = 51$$ with the same framework yields improved detail and spatial resolution demonstrating scalability. However errors are still evident over the ocean where unexpected temperatures are observed.

![City data]({{"/assets/images/Real data.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 7: Temperature mesurements in all cities on the globe over the month 07/2003 in degrees Celsisus*
{:.caption}

![City data lambda]({{"/assets/images/lambda L=51.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 8: Test errors for different folds of data*
{:.caption}

![City data reconstruction]({{"/assets/images/Real data.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 9: Resconstruction of the data in Figure 7 in degrees Celsius*
{:.caption}

---

## 6. Vector Field Reconstruction

To approximate divergence-free vector fields, we use vector spherical harmonics:

$$
\Phi^{\ell}_{m}(\vartheta, \varphi) = \mathbf{r} \times \nabla Y^{\ell}_{m}(\vartheta, \varphi).
$$

This guarantees $$\nabla \cdot \mathbf{u} = 0$$ by construction.

Approximations are made using a similar series expansion:

$$
\mathbf{u}(\vartheta, \varphi) = \sum_{\ell=0}^L \sum_{m=-\ell}^{\ell} v_{\ell m} \Phi^{\ell}_{m}(\vartheta, \varphi).
$$

The coefficients are fit using a similar regularised least squares procedure.

---

## 7. Real Wind Data Results

The method was applied to wind field observations at $$n = 100$$ and $$n = 500$$ sites. Reconstructions were inaccurate, especially with expected difficulties near the poles due to singularities and sparse data. 

![Wind data]({{"/assets/images/data n=100.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 10: $$n=100$$ data points from the wind data set.*
{:.caption}

![Wind data reconstruction]({{"/assets/images/streamplot n=100, L=50.png" | relative_url }}){: .img-fluid .border .shadow }
*Figure 11: Resconstruction of the data in Figure 10 in metres per second. Streamlines (Left), vector field magnitude (Right)*
{:.caption}

The results indicate that test error rises with increasing truncation degree $$L$$, suggesting potential model inaccuracies beyond just underfitting. Key reasons for these inaccuracies include:

Incompressibility Assumption:

- The 2D wind data from Meteostat may violate incompressibility because real wind has a vertical component (e.g., due to sloped terrain), which is ignored and data is given in 2D on the manifold.

- Terrain roughness and vegetation effects are unaccounted for, further deviating from incompressibility.

Data Quality Issues:

- Urban wind measurements are distorted by buildings and skyscrapers, introducing noise. Future work should exclude such data.

- Hourly averaged wind speed/direction data (like Meteostat’s) lose physical granularity, potentially skewing results.


---

## 8. Conclusion

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

### References

---
