# Inverse reconstruction of Einstein-cluster collapse and oscillations

Python code and numerical diagnostics accompanying the manuscript **“Inverse reconstruction of Einstein-cluster collapse and oscillations in tangential-pressure spacetimes”** (Jan J. Ostrowski).

## Contents

- `reconstruction_figures_refined.py` — main reproducibility script, generating both the asymptotic-collapse and long-lived oscillating examples, their figures, and shell-crossing diagnostics.
- `reconstruction_figures_refined.ipynb` — notebook counterpart.
- `DBG_diagnostics.csv` — sample numerical diagnostics from the manuscript calculation.
- `requirements.txt` — main Python dependencies. The notebook additionally requires a Jupyter environment.

The latest figure-generation code supersedes earlier development scripts. In particular, it uses the physical Jacobian

\[
D=\left(\frac{\partial R}{\partial M}\right)_t
 = \left(\frac{\partial R}{\partial M}\right)_\tau
   + \left(\frac{\partial R}{\partial\tau}\right)_M
     \left(\frac{\partial\tau}{\partial M}\right)_t,
\]

and computes the energy density as \(\rho=1/(4\pi R^2D)\). The oscillating initial density profile matches NFW only outside the inner transition region; it is not globally NFW.

## Running the calculations

Install Python 3 and the required packages:

```bash
python -m pip install -r requirements.txt
```

Then run from the repository's root folder:

```bash
python reconstruction_figures_refined.py
```

The script writes vector PDF figures into a `figures/` directory and produces `DBG_diagnostics.csv` and `DBG_summary.json` in the working directory. Runtime may be substantial because the script includes a long-term shell-crossing check and a higher-resolution comparison.

The model uses geometrized units \(G=c=M_{\rm tot}=1\). The long-lived oscillating example uses an Einasto equilibrium, a smoothly NFW-matched outer snapshot, and the time-coordinate correction needed for the physical shell-crossing diagnostic. Numerically detected crossing times refer to the specified discretized mass domain and do not prove continuum regularity.

## Reproducibility and citation

Please use a tagged repository release when citing a specific set of results. You can also connect the repository to Zenodo and archive a release to obtain a DOI. Add the journal citation and DOI here after publication.

## License

No open-source license is included in this upload template. The author should select and add an appropriate `LICENSE` before making reuse permissions explicit.
