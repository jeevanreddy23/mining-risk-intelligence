# Sources and Citations

## Dataset Sources

### WA Government and DMIRS
- `DMIRS-015` 1:500 000 State linear structures layer
  URL: https://catalogue.data.wa.gov.au/dataset/1-500-000-state-linear-structures-layer-dmirs-015
  Use in project: regional structural proxy, distance-to-structure features

- `DMIRS-038` Interpreted bedrock geology 1:100 000
  URL: https://catalogue.data.wa.gov.au/dataset/interpreted-bedrock-geology-1-100-000-dmirs-038
  Use in project: lithology and geology-unit context

- `DMIRS-070` Gravity (400m) of WA v1, 2020
  URL: https://catalogue.data.wa.gov.au/dataset/gravity-400m-of-wa-v1-2020-dmirs-070
  Use in project: gravity anomaly sampling and regional density-contrast proxy

- `DMIRS-046` Mineral exploration drillholes
  Source family: WA Data Catalogue / GeoVIEW WA public drillhole layers
  Use in project: drillhole density and subsurface exploration footprint

- `MINEDEX` Mines and Mineral Deposits
  URL: https://catalogue.data.wa.gov.au
  Use in project: mineral occurrence proximity and mining activity context

- `WAMEX` Mineral Exploration Reporting System
  URL: https://wamex.dmirs.wa.gov.au
  Use in project: drillhole evidence, lithology references, structural interpretations, geophysics reports

### Geoscience Australia
- `GA Earthquake Catalogue / event XML`
  URL: https://www.ga.gov.au/data-pubs/online-tools
  Use in project: regional seismicity context, event timing, magnitude, and depth

## Method References

### Seismicity
- Gutenberg, B., and Richter, C.F. (1944). Frequency of earthquakes in California. Bulletin of the Seismological Society of America, 34(4), 185-188.
  Use in project: synthetic magnitude-frequency behavior for prototype microseismic event generation

### Rock Mass Classification
- Bieniawski, Z.T. (1989). Engineering Rock Mass Classifications. Wiley.
  Use in project: prototype mapping of geology and structure proxies to rock-mass quality assumptions

- Hoek, E., Marinos, P., and Benissi, M. (1998). Applicability of the Geological Strength Index (GSI) classification for very weak and sheared rock masses. The case of the Athens Schist Formation. Bulletin of Engineering Geology and the Environment, 57, 151-160.
  Use in project: conceptual basis for inferred GSI-style proxy ranges

### Blast Vibration and Scaled Distance
- USBM RI 8507 and scaled-distance style blasting vibration relationships are used as prototype engineering logic only.
  Use in project: synthetic PPV generation based on charge weight and distance
  Note: current repository implementation uses a generic scaled-distance proxy and must be calibrated with mine data before operational use.

## Project Positioning
- Public datasets provide the `regional context layer`.
- Synthetic data is used only to complete the prototype `operational layer` until mine-site access is available.
- Production-grade blast vibration, PPV, microseismic, RQD, GSI, and label data must later be sourced from mine systems.
