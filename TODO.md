# TODO

Features used by the Belgium CoMix analysis (`mixres_analysis`, waves 1–8) that
were lost in the `AgeMixAgg` refactor.

1. **Soft upper-age boundary (`boundary_tau`).** Extend the latent age grid by
   two ages (max+1, max+2) and add
   `numpyro.factor(-0.5 * sum((m / tau)**2))` on the baseline contact marginals
   `m` at those ages (contacts per respondent, summed over the original
   contact-age support, before the fatigue effect). This is a strong shrinkage
   towards zero, not an observed zero count. The likelihood, population
   normalisation and standard outputs stay on the original grid. The analysis
   used `tau = 0.5`.
2. **Configurable `conditional_intensity` smoothing strength.** `Allocation`
   fixes the pseudo-count at one pooled pseudo-contact per respondent age
   (`n[a, d] + n[d] / sum_d n[d]`). Expose the total pseudo-contacts as a
   parameter; the analysis compared 0.1, 1 and 10.
3. **Public SVI result accessor.** `ContactModel` stores the SVI result only as
   `_svi_result`; downstream code needs `params` and `losses`.
