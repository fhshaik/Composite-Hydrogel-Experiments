# Figures/Raw Graphs: Hydrogels Compression Analysis

This folder contains raw and processed plots from hydrogel compression experiments.

## Folder Structure
- Each date has its own folder.
- Within each date, plots are separated by test type ("sample" for hydrogel, "air" for control/air tests).
- Filenames include the trial number and a label for the plot type.

## Plot Types
### stress_strain
- **What it shows:** Stress vs. strain for a single hydrogel sample.
- **Red curve:** Raw stress (from force sensor).
- **Black curve:** Adjusted stress (after subtracting air/control background).
- **Meaning:** Shows how the hydrogel resists compression, and how much of the force is due to the sample vs. the impactor/air.

### air_force_cut
- **What it shows:** Force (or stress) vs. depth for an air/control test (no sample).
- **Meaning:** Used to correct for the force due to the impactor itself, so that sample measurements are accurate.

## How to Use
- Compare "stress_strain" plots across trials and dates to see how different hydrogels behave.
- Use "air_force_cut" plots to check the background/impactor force profile.
- All plots are saved as both PNG and SVG for publication or further analysis.
