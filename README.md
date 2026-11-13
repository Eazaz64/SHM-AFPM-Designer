# AFPM-BLDC SHM Designer

Offline Windows program for the **Space Harmonic Method (SHM)** design of axial flux permanent
magnet BLDC motors (single-stator single-rotor **SSSR** and single-stator double-rotor **SSDR**)
driven by a six-step controller. It is a Python port of the MATLAB code of Chapter 4
(`AFPM_params`, `carter_coeff`, `shm_Bn`, `winding_factor`, `shm_backemf`, `flux_check`,
`circuit_params`, `performance`, and the speed-torque / efficiency figures).

The program works fully offline. Nothing is sent over the network.

## What it calculates

Sizing and shear stress, Carter coefficient, air-gap flux density (all harmonics), winding factors,
phase and line back-EMF, THD, **Ke** and **Kt**, resistance and inductance, tooth / yoke / back-iron
flux density, rated current, torque, power, losses, efficiency, voltage check, active mass,
torque density, and the speed-torque characteristic at the chosen DC bus voltage.

## Graphs (each one: save as PNG / SVG / PDF, and save its data as CSV)

1. Back-EMF waveform (phase and line, 60 degree conduction window)
2. Air-gap flux density at the inner, mean and outer radius
3. Harmonic spectrum of the air-gap flux density
4. Harmonic spectrum of the back-EMF (phase and line, THD)
5. Winding factor of each harmonic
6. Star of slots
7. **Torque versus DC-link current (Kt)**
8. **Back-EMF versus speed (Ke)** - 60 degree average basis, peak basis and phase fundamental
9. Speed-torque characteristic at the DC bus voltage (current limit and voltage limit)
10. Output power and efficiency versus torque
11. Efficiency map
12. EMF-shape torque ripple
13. Tooth flux density along the radius
14. Losses at the rated point

Your own measurements can be overlaid on graphs 8 and 9 (menu *Overlay*): a CSV or text file
with two numeric columns, x then y (a header row is allowed).

**File > Export everything** writes `results.csv`, `inputs.csv`, `design.json` and every graph
(PNG, 300 dpi) with its CSV into one folder. CSV files open directly in Excel.

## Get the Windows .exe from GitHub

1. Create a new repository on GitHub and upload all files of this folder (keep the folder
   structure, including `.github/workflows/build-windows.yml`).
   **If the `.github` folder did not upload** (folders starting with a dot are hidden and are
   skipped by the GitHub drag-and-drop upload): on the repository page press *Add file > Create new
   file*, type `.github/workflows/build-windows.yml` as the file name (typing the slashes creates the
   folders), paste the contents of `build-windows.yml.txt` from this project, and press *Commit*.
2. Open the **Actions** tab. The workflow *Build Windows app* runs by itself on every push
   (or press *Run workflow*). It runs the tests, builds the .exe with PyInstaller, and starts the
   built .exe once with `--selftest` to check that the window opens and every graph exports.
3. Download `AFPM-SHM-Designer.exe` from the finished run (*Artifacts*).
4. For a permanent download link, create a tag: `git tag v1.0.0` and `git push origin v1.0.0`.
   The .exe is then attached to a release under **Releases**.

The first start of the one-file .exe takes a few seconds (it unpacks itself). Windows SmartScreen
may show "unknown publisher" because the file is not code-signed: choose *More info > Run anyway*.
If an antivirus program complains about the one-file build, replace `--onefile` with `--onedir`
in the workflow and distribute the folder as a zip.

## Run from source (Windows, Python 3.10 or newer from python.org)

```
pip install -r requirements.txt
python run_app.py
```

Tests (no screen needed): `pip install pytest` then `python -m pytest -q tests`.

## How it was checked

`tests/test_model.py` compares the Python results with the expected output printed in the MATLAB
README for both machines (winding factors, back-EMF, THD, Ke, R, L, flux densities, current,
torque, power, losses, efficiency, voltage, mass). All agree to the printed precision. One
difference: the EMF-shape torque ripple is 15.146 %, which the MATLAB README lists as 15.2 %
(a last-digit rounding of the same number). `tests/test_gui_logic.py` runs every button handler.

## Differences from the MATLAB code (same results for the thesis machines)

- The coil sequence is generated from the star of slots for any Q and p. For 12 slots and 7 pole
  pairs it gives the same sequence A C -C -B B A -A -C C B -B -A as `AFPM_params.m`.
- Turns per phase = (Q / 3) x turns per coil (4 x 16 = 64 for the prototypes).
- The core-loss sums are written for any Steinmetz exponent (the MATLAB maps assume 2).
- The efficiency map is limited by the DC bus voltage and by the controller current limit.
- Optional calibration factors for Ke/Kt and inductance (default 1.0 = pure SHM).

## Limits of the model

The SHM is a 2D-slice model. It neglects 3D end effects and fringing at the inner and outer radii,
and the speed-torque curve is an average model that ignores inductance and commutation delay.
Compare the predictions with measurements before relying on them, and use the calibration factors
only with measured evidence.

Field model after Taghipour Boroujeni et al., IET Electric Power Applications, 2016.
