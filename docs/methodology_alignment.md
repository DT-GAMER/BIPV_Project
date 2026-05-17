# Methodology Alignment

This project follows the facade BIPV methodology described in the referenced
workflow figures:

```text
Facade RGB image
  -> Grounding DINO object detection
  -> SAM mask generation
  -> Stable Diffusion / LaMa inpainting
  -> perspective transformation
  -> facade alignment
  -> facade parsing: wall/window/obstacle masks
  -> shadow and irradiance analysis
  -> BIPV output estimation
```

## Current Implementation

The current codebase implements the RGB-image facade parsing branch:

| Methodology stage | Code module | Status |
| --- | --- | --- |
| Image acquisition and preprocessing | `src/preprocessing.py` | implemented |
| Grounding DINO detection | `src/detection.py` | implemented |
| SAM mask generation | `src/segmentation.py`, `src/inpainting.py` | implemented |
| Obstacle mask expansion | `src/inpainting.py` | implemented |
| Obstacle removal / inpainting | `src/inpainting.py` | implemented with TELEA, LaMa, optional Stable Diffusion |
| Perspective transformation | `src/geometry.py` | implemented |
| Facade alignment/grid structuring | `src/alignment.py` | implemented baseline |
| Window/wall/facade parsing | `src/segmentation.py`, `src/bipv_segmentation.py` | implemented baseline |
| Real-world scale estimation | `src/scale_estimation.py`, `src/scaling.py` | implemented as automatic estimate |
| Area and capacity estimation | `src/area.py`, `src/energy.py` | implemented baseline |
| Export | `src/export.py` | implemented |
| Workflow visualization | `src/visualization.py` | implemented |
| Batch processing | `src/batch.py` | implemented |

## Research Gap Still To Implement

The paper methodology includes data sources and simulation layers that are not
fully implemented yet:

1. **Building footprint data**
   - Target sources: Mapbox, OpenStreetMap, city GIS, Microsoft building footprints.
   - Purpose: provide building outline, orientation, and urban context.

2. **3D building model**
   - Requires footprint plus building height.
   - Purpose: align facade image to actual building geometry.

3. **Shadow casting from neighbouring buildings**
   - Paper uses 3D shadow modelling such as `pybdshadow`.
   - Current code only has image-based shadow detection.

4. **Meteorological data**
   - Target sources: NSRDB, PVGIS, Meteostat, local TMY files.
   - Required variables: DNI, DHI, GHI, temperature, humidity, wind speed.

5. **PV simulation**
   - Target library: `pvlib`.
   - Current code estimates capacity with area and panel wattage only.
   - Future implementation should estimate irradiance, cell temperature, DC power,
     and annual/monthly/daily energy yield.

## Important Scientific Framing

For uploaded RGB images with no geolocation or footprint metadata, the system can
automatically estimate facade BIPV potential, but absolute metre-square values
are inferred, not physically measured. Accuracy improves when one or more of the
following are available:

- geolocation,
- building footprint,
- building height,
- facade orientation,
- Google Earth validation dimensions,
- camera metadata,
- known floor count,
- meteorological data.

## Target Full System

The final system should support two modes:

### Mode A: RGB-Only Automatic Estimate

User uploads image only.

```text
image -> facade parsing -> automatic scale estimate -> usable area -> capacity
```

Output label:

```text
estimated usable BIPV area
```

### Mode B: Geospatial Engineering Estimate

User provides image plus geolocation or building id.

```text
image + footprint + height + weather
  -> facade parsing
  -> 3D alignment
  -> shadow simulation
  -> pvlib energy simulation
  -> validated BIPV output
```

Output label:

```text
engineering BIPV potential estimate
```

## Next Engineering Milestones

1. Add `src/footprints.py` for footprint loading and facade orientation.
2. Add `src/weather.py` for meteorological data ingestion.
3. Expand `src/energy.py` to use `pvlib` irradiance and temperature models.
4. Add `src/shadow_3d.py` for geometry-based shadow simulation.
5. Add validation dataset format:

```text
data/validation/
  cases.csv
  images/
  masks/
  google_earth_measurements.csv
```

6. Add metrics:

```text
facade area error %
usable area error %
window mask IoU
shadow mask IoU
capacity error %
annual energy error %
```
