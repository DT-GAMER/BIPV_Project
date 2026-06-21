# Calculation Breakdown for BIPV Facade Area Results

## Project Title

**An image-based pipeline for BIPV facade area estimation, converting street-level facade images into PVsyst-ready data.**

This document explains how the numerical results from the batch analysis were obtained. It is written as a defense/presentation guide so the calculation logic can be explained clearly to supervisors, examiners, or project stakeholders.

## 1. What the Pipeline Calculates

For each street-level building image, the pipeline estimates:

- Number of floors
- Real-world facade height in metres
- Real-world facade width in metres
- Total facade area in square metres
- Usable BIPV facade area in square metres
- Percentage of facade usable for BIPV
- Percentage of facade excluded from BIPV
- Number of PV panels that can fit
- Installed PV capacity in kWp
- Estimated annual energy output in kWh
- JSON and Excel outputs for PVsyst-ready downstream use

The image-processing stages are:

1. Facade image input
2. Obstacle detection
3. Obstacle removal / inpainting
4. Facade alignment / rectification
5. Facade segmentation result
6. Area and PV capacity estimation

## 2. How the Number of Floors Is Obtained

The model detects windows and opening rows on the facade. These detected openings are grouped vertically into floor bands.

For example, if the detected window rows form four clear horizontal bands, the model estimates:

```text
num_floors = 4
```

The result also records where the floor count came from:

```text
floor_count_source
```

Typical values include:

```text
dino_window_boxes
segmented_window_mask
segmented_window_mask_with_facade_extent
```

This means the model can count floors either from direct Grounding DINO window detections or from the final segmented window/opening mask.

## 3. How Facade Height Is Calculated

The real-world facade height is estimated using:

```text
Facade height (m) = Number of floors × Estimated floor height (m)
```

The floor height is an architectural prior. In this project, the default urban value is usually:

```text
3.3 m per floor
```

For buildings with smaller windows or high-rise characteristics, the model may use:

```text
3.4 m per floor
```

For low-rise buildings with larger window proportions, it may use:

```text
3.1 m per floor
```

Example for `IMG_2000`:

```text
Number of floors = 4
Estimated floor height = 3.3 m

Facade height = 4 × 3.3
              = 13.2 m
```

Example for `IMG_0000`:

```text
Number of floors = 15
Estimated floor height = 3.4 m

Facade height = 15 × 3.4
              = 51.0 m
```

## 4. How Pixel-to-Metre Scaling Is Obtained

After facade segmentation, the model knows the facade height in pixels:

```text
facade_height_px
```

It also estimates the real-world height in metres:

```text
height_m
```

The pixel-to-metre scale is:

```text
pixels_per_meter = facade_height_px / height_m
```

This tells the model how many image pixels represent one metre in real-world facade height.

Example for `IMG_2000`:

```text
pixels_per_meter = 35.1515 px/m
```

This means approximately 35.15 pixels in the rectified facade image represent 1 metre in real life.

## 5. How Facade Width Is Calculated

Once the model knows the pixel-to-metre scale, it converts facade width from pixels into metres:

```text
Facade width (m) = facade_width_px / pixels_per_meter
```

The `facade_width_px` is obtained from the width of the segmented facade mask after rectification.

For example, `IMG_2000` produced:

```text
Facade width = 28.16 m
```

This means the segmented, rectified facade was estimated to represent a real-world width of approximately 28.16 metres.

## 6. How Total Facade Area Is Calculated

The total facade area is calculated using the standard rectangle area equation:

```text
Total facade area (m²) = Facade height (m) × Facade width (m)
```

Example for `IMG_2000`:

```text
Height = 13.20 m
Width  = 28.16 m

Total facade area = 13.20 × 28.16
                  = 371.76 m²
```

Example for `IMG_2001`:

```text
Height = 19.80 m
Width  = 34.34 m

Total facade area = 19.80 × 34.34
                  = 679.84 m²
```

## 7. How the Mask Pixel Count Is Used

The segmentation output is a binary mask. Each pixel is either:

```text
True  = belongs to the region
False = does not belong to the region
```

In Python, `True` behaves like `1` and `False` behaves like `0`.

Therefore, the model counts pixels by summing the mask:

```python
facade_area_px = facade_mask.sum()
usable_area_px = usable_mask.sum()
```

The facade mask contains all pixels classified as facade wall surface. The usable mask contains only the pixels that remain after excluding windows, doors, balconies, roof/opening areas, and other unavailable regions.

## 8. How Pixels Are Converted to Square Metres

The model first calculates how much real-world area one facade pixel represents:

```text
m² per pixel = total facade area (m²) / facade area pixel count
```

In code this is:

```python
px_to_m2 = dimensions["total_facade_area_m2"] / facade_area_px
```

Then the usable BIPV area is:

```text
Usable area (m²) = usable mask pixel count × m² per pixel
```

In code this is:

```python
usable_area_m2 = usable_mask.sum() * px_to_m2
```

This means the model does not guess the usable area directly. It:

1. Segments the facade.
2. Counts facade pixels.
3. Converts the total facade into real-world square metres.
4. Calculates the square-metre value of each pixel.
5. Counts usable pixels.
6. Converts usable pixels into square metres.

## 9. What Is Excluded From Usable BIPV Area

The usable BIPV area is the facade wall area remaining after exclusions.

The excluded regions include:

- Windows
- Doors
- Balconies
- Roof/opening masks where detected
- Architectural voids or unavailable facade regions
- Optional obstacle exclusion if enabled

The general logic is:

```text
usable_mask = facade_mask
              - window_mask
              - door_mask
              - balcony_mask
              - roof/opening mask
```

The code also dilates some exclusions slightly to represent mounting clearance around openings.

## 10. How Usable and Excluded Percentages Are Calculated

Usable percentage:

```text
Usable area (%) = usable area (m²) / total facade area (m²) × 100
```

Excluded area:

```text
Excluded area (m²) = total facade area (m²) - usable area (m²)
```

Excluded percentage:

```text
Excluded area (%) = 100 - usable area (%)
```

Example for `IMG_2000`:

```text
Total facade area = 371.76 m²
Usable area       = 252.89 m²

Usable percentage = 252.89 / 371.76 × 100
                  = 68.03%

Excluded area     = 371.76 - 252.89
                  = 118.87 m²

Excluded percent  = 100 - 68.03
                  = 31.97%
```

## 11. How Panel Count Is Calculated

The project assumes:

```text
Panel area = 1.7 m²
Panel power = 350 W
```

The number of panels is calculated as:

```text
Number of panels = floor(usable area / panel area)
```

The `floor` operation means only complete panels are counted.

Example for `IMG_2000`:

```text
Usable area = 252.89 m²
Panel area  = 1.7 m²

Panels = floor(252.89 / 1.7)
       = floor(148.76)
       = 148 panels
```

## 12. How Installed Capacity Is Calculated

Each panel is assumed to be:

```text
350 W = 0.35 kW
```

Installed capacity is:

```text
Total capacity (kW) = number of panels × watts per panel / 1000
```

Example for `IMG_2000`:

```text
Panels = 148
Watts per panel = 350 W

Capacity = 148 × 350 / 1000
         = 51.8 kW
```

## 13. How Annual Energy Output Is Calculated

The project uses a specific yield assumption:

```text
Specific yield = 950 kWh/kWp/year
```

Annual energy is calculated as:

```text
Annual energy (kWh) = installed capacity (kWp) × specific yield
```

Example for `IMG_2000`:

```text
Installed capacity = 51.8 kWp
Specific yield = 950 kWh/kWp/year

Annual energy = 51.8 × 950
              = 49,210 kWh/year
```

This is a simplified PV yield estimate. It is useful for screening and comparison, but a full PVsyst simulation would later refine it using orientation, climate data, shading, system losses, and module parameters.

## 14. Batch Result Table

| Building | Floors | Height (m) | Width (m) | Total facade area (m²) | Segmented usable area (m²) | Usable area (%) | Excluded area (m²) | Excluded area (%) | Panels | Capacity (kWp) | Annual energy (kWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMG_2000 | 4 | 13.20 | 28.16 | 371.76 | 252.89 | 68.03 | 118.87 | 31.97 | 148 | 51.80 | 49,210.0 |
| IMG_2001 | 6 | 19.80 | 34.34 | 679.84 | 519.16 | 76.36 | 160.68 | 23.64 | 305 | 106.75 | 101,412.5 |
| IMG_2222 | 4 | 13.20 | 21.48 | 283.51 | 216.92 | 76.51 | 66.59 | 23.49 | 127 | 44.45 | 42,227.5 |
| IMG_6387 | 4 | 12.40 | 18.19 | 225.54 | 179.43 | 79.55 | 46.11 | 20.45 | 105 | 36.75 | 34,912.5 |
| IMG_0000 | 15 | 51.00 | 31.01 | 1,581.34 | 1,395.60 | 88.25 | 185.74 | 11.75 | 820 | 287.00 | 272,650.0 |
| **Total** | — | — | — | **3,141.99** | **2,564.00** | **81.60** | **578.00** | **18.40** | **1,505** | **526.75** | **500,412.5** |

## 15. Worked Example: IMG_2000

### Step 1: Floor count

```text
Detected floors = 4
```

### Step 2: Facade height

```text
Floor height prior = 3.3 m

Height = 4 × 3.3
       = 13.2 m
```

### Step 3: Facade width

```text
Width = 28.16 m
```

This was obtained from the rectified facade mask width and the pixel-to-metre scale.

### Step 4: Total facade area

```text
Area = 13.2 × 28.16
     = 371.76 m²
```

### Step 5: Usable area

```text
Usable area = 252.89 m²
```

This comes from counting the usable pixels in the final BIPV mask and converting them to square metres.

### Step 6: Usable percentage

```text
Usable percentage = 252.89 / 371.76 × 100
                  = 68.03%
```

### Step 7: Excluded area

```text
Excluded area = 371.76 - 252.89
              = 118.87 m²
```

### Step 8: Panel count

```text
Panel count = floor(252.89 / 1.7)
            = 148 panels
```

### Step 9: Installed capacity

```text
Capacity = 148 × 350 / 1000
         = 51.8 kWp
```

### Step 10: Annual energy

```text
Annual energy = 51.8 × 950
              = 49,210 kWh/year
```

## 16. Important Defense Explanation

If asked how the model knows the real-world size, the answer is:

The model estimates real-world scale from the detected number of floors and an architectural floor-height prior. It then uses the rectified facade mask to calculate facade width and pixel-to-square-metre conversion. If Google Earth dimensions are supplied, the model can use those dimensions directly as the calibration reference.

If asked whether the result is a direct physical measurement, the answer is:

No. In automatic mode, it is an image-based engineering estimate. It becomes metrically stronger when calibrated with Google Earth or measured facade dimensions. The segmentation determines the usable fraction, while the scale estimation converts that fraction into square metres.

If asked why the usable area is lower than the total facade area, the answer is:

Because windows, doors, balconies, roof/opening areas, and unavailable facade regions are excluded from the installable BIPV surface. The usable area represents the remaining facade wall surface where BIPV modules could potentially be installed.

If asked why panel count is lower than usable area divided exactly by panel area, the answer is:

The panel count uses only complete panels:

```text
floor(usable area / panel area)
```

Partial panels are not counted.

## 17. Limitations to State Clearly

The current results are suitable for preliminary BIPV facade screening. However, the following limitations should be acknowledged:

- Automatic scale estimation depends on the assumed floor height.
- Strong perspective distortion can affect facade width estimation.
- Missed or false window segmentation affects usable area.
- Obstacle reconstruction may not perfectly recover hidden architecture.
- Annual energy is estimated using a fixed specific yield, not full irradiance simulation.
- Full PVsyst validation should use measured facade dimensions, orientation, shading, and local meteorological data.

## 18. Summary Statement

The pipeline converts a street-level building image into structured BIPV-ready outputs by detecting the facade, removing obstacles, rectifying the facade, segmenting usable and unusable regions, converting mask pixels into square metres, and estimating PV panel capacity and annual energy yield. The calculations are based on explicit geometric formulas and binary mask pixel counts, making the process explainable and reproducible.
