# Development Process and Geospatial Scaling Methodology

## Project Title

**An image-based pipeline for BIPV facade area estimation, converting street-level facade images into PVsyst-ready data.**

This project develops an image-based workflow for estimating usable building-integrated photovoltaic (BIPV) facade area from ordinary street-level building images. The system combines computer vision, image reconstruction, facade rectification, semantic facade parsing, geocoding, open geospatial building footprints, and metric area conversion. The final outputs are engineering-ready values and files, including usable facade area, facade element segmentation, PV capacity estimates, JSON export, and Excel export for PVsyst-style analysis.

## Why The Pipeline Was Developed

The original problem was that manual facade measurement is slow, inconsistent, and difficult to scale across many buildings. A user should ideally upload a building image and, where available, provide the building address or coordinate. The system then performs the major analysis stages automatically:

1. Identify the building facade in the image.
2. Detect foreground obstacles such as vehicles, trees, poles, fences, shrubs, and signs.
3. Remove or mask obstacles so they do not corrupt facade segmentation.
4. Rectify the facade while preserving the original image size.
5. Segment facade elements such as wall, windows, doors, balconies, and roof edges.
6. Convert image pixels into real-world metres.
7. Calculate usable BIPV facade area.
8. Estimate panel count, capacity, and annual energy.
9. Export JSON and Excel outputs.

The key development challenge was metric scaling. A photo alone does not contain true physical dimensions unless it is calibrated. The current solution uses three scaling routes:

1. **Geospatial scaling** using Google Geocoding plus Overture Maps or OSM building footprints.
2. **Clicked facade-line scaling** where the user draws the exact facade width on a map.
3. **Image-only fallback scaling** using detected floor count and facade aspect when no geospatial reference is available.

## High-Level Pipeline

The final user-facing workflow is:

```text
Street-level facade image
        ↓
Image acquisition and preprocessing
        ↓
Grounding DINO obstacle/facade detection
        ↓
SAM obstacle segmentation
        ↓
Obstacle masking and inpainting
        ↓
Facade rectification and alignment
        ↓
Facade element segmentation
        ↓
Geospatial / clicked-line / image-only scaling
        ↓
Usable BIPV area estimation
        ↓
Panel capacity and energy estimate
        ↓
JSON and Excel export
```

## Core Code Modules

The main modules are:

- `src/pipeline.py`: Orchestrates the full end-to-end analysis.
- `src/config.py`: Stores all configuration options used by the pipeline.
- `src/geocoding.py`: Resolves a user address to latitude/longitude using Google Geocoding.
- `src/geospatial.py`: Fetches Overture/OSM building footprints, measures edges, creates verification maps, and supports clicked facade-line measurement.
- `src/colab_facade_line.py`: Creates a Colab map widget so the user can draw the exact facade edge.
- `src/scaling.py`: Converts image facade pixels into metric width, height, area, and pixel-to-metre scale.
- `src/area.py`: Converts facade masks into usable BIPV area and panel capacity.
- `src/geometry.py`: Performs facade rectification and validates whether the rectification improves alignment.
- `src/segmentation.py`: Produces facade, window/opening, door, balcony, and roof masks.
- `src/export.py`: Saves JSON and Excel engineering outputs.

## Role Of Google Geocoding API

Google Geocoding is used only to locate the building. It does **not** directly calculate facade width, height, or area.

Input:

```text
114 Dundee St, Fountainbridge, Edinburgh EH11 1AB, United Kingdom
```

Output:

```text
latitude
longitude
formatted address
Google place ID
source = google-geocoding
```

The code path is:

```text
src/geocoding.py
resolve_coordinates()
    → geocode_google()
```

If the user provides exact latitude and longitude, the system skips Google Geocoding and uses those coordinates directly.

## Role Of Overture Maps

Overture Maps provides open building footprint data. A footprint is the top-down polygon shape of the building. It is not a facade image; it is a plan-view geometry.

The system queries Overture around the geocoded coordinate and retrieves nearby building polygons. For each building polygon, the system extracts:

- Overture building ID.
- Building footprint points.
- Footprint edges.
- Edge lengths in metres.
- Edge bearings.
- Footprint centroid.
- Distance from the input address/coordinate point.
- Available tags such as building class, subtype, height, or levels if present.

The code path is:

```text
src/geospatial.py
fetch_geospatial_building_reference()
    → fetch_overture_building_reference()
```

Overture is tried first. If Overture fails or no building is found, the system falls back to OSM/Overpass:

```text
fetch_geospatial_building_reference()
    → fetch_osm_building_reference()
```

## How The Building ID Is Gotten

When Overture returns a building feature, the building ID is read from the Overture feature properties:

```python
properties.get("id") or feature.get("id")
```

The output appears as:

```text
overture_id: 51760847-5793-4486-aaf9-76db99c95025
source: overture-maps
```

If OSM/Overpass is used instead, the output contains:

```text
osm_id
osm_type
source: osm-overpass
```

This ID helps verify which geospatial building polygon was selected. It does not come from the image; it comes from the geospatial dataset.

## How The Correct Building Is Selected

After the address is geocoded, the system queries nearby building footprints within a search radius, usually 60 metres.

For each candidate building:

1. The building footprint polygon is extracted.
2. The polygon centroid is calculated.
3. The distance from the geocoded point to the centroid is calculated.
4. Candidates are sorted by distance.
5. The nearest footprint is selected.

This is why visual verification is important. Sometimes the address point is on the road or at a doorway, and a large building block may be selected instead of the exact visible facade. The project therefore includes a verification map.

## How The Width Is Gotten

There are three width modes.

### Mode 1: Clicked Facade-Line Width

This is the most defensible mode for conference/paper presentation because the user explicitly selects the exact photographed facade edge on the map.

In Colab, the user opens the facade-line selector and draws a red line along the facade frontage. The system reads the two endpoint coordinates:

```text
start_lat, start_lon
end_lat, end_lon
```

Then it calculates the real-world distance between those points using the haversine formula:

```text
a = sin²(Δφ / 2) + cos(φ1) cos(φ2) sin²(Δλ / 2)
d = 2R atan2(√a, √(1 − a))
```

Where:

- `φ1`, `φ2` are latitudes in radians.
- `Δφ` is the latitude difference.
- `Δλ` is the longitude difference.
- `R` is Earth radius, 6,371,008.8 m.
- `d` is facade width in metres.

The output then records:

```text
facade_width_source: user-selected-map-line
facade_width_m: measured line distance
clicked_facade_line:
    start
    end
    width_m
    bearing_deg
```

This mode avoids relying on automatic edge selection when the building footprint is complex.

### Mode 2: Automatic Overture/OSM Edge Width

If the user does not draw a line, the system measures every edge of the selected footprint polygon.

Each footprint edge is measured using the same haversine distance formula. The first-pass rule is:

- If a facade bearing is supplied, choose the edge whose bearing best matches it.
- Otherwise, choose a candidate edge using edge length, image-estimated width, and proximity to the query point.

The output may show:

```text
facade_width_source: overture-maps-image-width-matched-edge
```

This means the system did not simply choose the longest edge. It compared the measured footprint edges against the image-implied facade width and selected the edge that better matches the visible facade.

### Mode 3: Image-Only Width Fallback

If no address, coordinate, clicked line, Overture, or OSM footprint is available, the system falls back to image-only scaling.

In this mode:

```text
height_m = detected_floors × assumed_floor_height_m
pixels_per_meter = facade_height_px / height_m
width_m = facade_width_px / pixels_per_meter
```

This is useful when no geospatial data exists, but it is less defensible because it depends on floor detection and assumed floor height.

## How The Height Is Gotten

Height is calculated using a priority order.

### Priority 1: Geospatial Height Tag

If Overture or OSM provides a building height tag, the system uses it directly:

```text
height_m = geospatial_reference["height_m"]
height_method = geospatial-source-height
```

If the dataset provides building levels instead of height:

```text
height_m = building_levels × default_floor_height_m
```

However, many Overture/OSM buildings do not contain height or level tags. In the example output, this is why:

```text
height_m: None
height_source: None
building_levels: None
```

appears inside `geospatial_reference`.

### Priority 2: Geospatial Width Plus Facade Pixel Aspect

When geospatial height is missing, the current improved method does **not** rely mainly on floor count. Instead, it uses the measured geospatial width and the visible facade aspect ratio from the image mask.

The formula is:

```text
facade_aspect = facade_height_px / facade_width_px
height_m = measured_width_m × facade_aspect
```

Example from the output:

```text
facade_height_px = 506
facade_width_px = 932
facade_aspect_h_over_w = 0.542918
reference_width_m = 29.7854
height_m = 29.7854 × 0.542918 = 16.1711 m
```

The output records:

```text
height_source: facade-aspect-from-geospatial-width
height_method: geospatial-width-facade-pixel-aspect
```

This is better than pure floor counting because it uses an externally measured width and the actual visible facade geometry.

### Priority 3: Floor Count Fallback

If no geospatial width is available, the system estimates height from floor count:

```text
height_m = num_floors × floor_height_m
```

The floor count is estimated from window/opening rows and facade extent. The output keeps this value even when geospatial scaling is used:

```text
floor_count_height_estimate_m
```

This is now mainly a fallback and comparison value.

## How Pixel-To-Metre Conversion Works

The facade segmentation produces a binary facade mask. In a binary mask:

- `True` or white pixels represent facade surface.
- `False` or black pixels represent non-facade/background.

The system finds the facade mask bounding extent:

```text
facade_width_px
facade_height_px
```

Then:

```text
pixels_per_meter_x = facade_width_px / width_m
pixels_per_meter_y = facade_height_px / height_m
```

When height is derived from geospatial width and image aspect, both scale values become consistent:

```text
pixels_per_meter_x ≈ pixels_per_meter_y
```

For area conversion:

```text
total_facade_area_m2 = width_m × height_m
facade_area_px = count of True pixels in facade_mask
px_to_m2 = total_facade_area_m2 / facade_area_px
```

So each facade mask pixel receives a real-world area value.

## How Usable BIPV Area Is Calculated

The usable area starts from the facade wall mask:

```text
usable_mask = facade_mask
```

Then the pipeline removes unusable facade elements:

```text
usable_mask = facade_mask
              − windows
              − doors
              − balconies
              − roof/non-wall exclusions
```

In code, windows, doors, and balconies are slightly dilated before subtraction so that a small mounting clearance around openings is also excluded.

The usable area is then:

```text
usable_area_m2 = usable_pixel_count × px_to_m2
```

Where:

```text
usable_pixel_count = count of True pixels in usable_mask
```

## How Panel Capacity Is Calculated

The current engineering assumptions are:

```text
panel_area_m2 = 1.7
watts_per_panel = 350
panel_efficiency = 0.20
```

Panel count:

```text
num_panels = floor(usable_area_m2 / panel_area_m2)
```

Total capacity:

```text
total_capacity_kw = num_panels × watts_per_panel / 1000
```

For example:

```text
usable_area_m2 = 358.62
num_panels = floor(358.62 / 1.7) = 210
total_capacity_kw = 210 × 350 / 1000 = 73.5 kW
```

## How Annual Energy Is Calculated

The simplified energy estimate uses a specific yield:

```text
specific_yield_kwh_per_kwp = 950
```

Then:

```text
annual_kwh = total_capacity_kw × specific_yield_kwh_per_kwp × (1 − shading_loss_fraction)
```

Shadow analysis is currently disabled in the focused pipeline, so:

```text
shading_loss_fraction = 0
```

Therefore:

```text
annual_kwh = total_capacity_kw × 950
```

Example:

```text
annual_kwh = 73.5 × 950 = 69,825 kWh
```

## What The Geospatial Verification Map Shows

The generated HTML map is used to prove what building and facade edge were used.

It displays:

- Building footprint polygon.
- Red selected facade edge.
- Green footprint centroid.
- Orange input address/coordinate point.
- Building source, ID, width, edge source, and distance from the input point.

Important fields:

```text
source
overture_id or osm_id
facade_width_m
facade_width_source
distance_to_query_m
footprint_edges
clicked_facade_line
```

If the red line is on the wrong side of the building, the width may be wrong for the photographed facade. In that case, the clicked facade-line method should be used.

## Recommended User Flow In Colab

### Case A: User Has Image And Address

1. Upload or select the facade image.
2. Enter the building address.
3. Store the Google Maps API key in Colab Secrets as `GOOGLE_MAPS_API_KEY`.
4. Run geocoding to get latitude/longitude.
5. Open the map selector.
6. Draw a line along the exact photographed facade edge.
7. Run the BIPV pipeline.
8. Verify that:

```text
scale_source = geospatial
facade_width_source = user-selected-map-line
height_method = geospatial-width-facade-pixel-aspect
clicked_facade_line_used = True
```

### Case B: User Has Image And Coordinates

1. Upload or select the facade image.
2. Enter latitude and longitude.
3. Skip Google Geocoding.
4. Draw or automatically select the facade edge.
5. Run the BIPV pipeline.

### Case C: User Has Image Only

1. Upload or select the facade image.
2. Leave address and coordinates empty.
3. The pipeline falls back to automatic image scale estimation.
4. Output should be treated as an estimate, not a geospatially validated measurement.

Expected output:

```text
scale_source = automatic-image-scale-estimate
validation.status = estimated-no-reference
```

## How To Explain The Method In A Defense

A clear defense explanation is:

> The image alone provides facade geometry in pixels, but not metric dimensions. Therefore, the system first uses Google Geocoding to convert the building address into coordinates. It then queries Overture Maps or OSM to retrieve the building footprint. The footprint gives real-world plan geometry and edge lengths. For the most reliable case, the exact photographed facade edge is selected on the map, and its length is calculated from the two endpoint coordinates using the haversine distance formula. This gives the facade width in metres. Since most open geospatial data does not include building height, the system estimates height from the visible facade aspect ratio in the segmented image: height equals measured width multiplied by facade pixel height divided by facade pixel width. The resulting width and height give total facade area, and the segmentation mask is used to count which pixels are usable after excluding windows, doors, balconies, and roof/non-wall regions. Finally, pixel counts are converted to square metres and used to estimate module count, capacity, and annual energy output.

## Limitations And Assumptions

The current method is strong, but the following limitations must be stated clearly:

1. Google Geocoding gives a location point, not a facade dimension.
2. Overture/OSM footprints are plan-view building outlines, not street-view facade models.
3. Some Overture footprints represent a whole building block, so the automatic edge may not always be the photographed facade.
4. The clicked facade-line method is the most reliable width calibration currently available.
5. Height is only directly measured if Overture/OSM provides height or building levels.
6. When height metadata is missing, height is estimated from measured width and image facade aspect ratio.
7. This height estimate assumes the detected facade mask corresponds to the same facade edge measured on the map.
8. Severe perspective distortion, partial facade visibility, or poor facade segmentation can affect height and area.
9. The image-only fallback should be labelled as an estimate because it depends on floor count and assumed floor height.

## Final Current Method Summary

The current best method is:

```text
Image + address/coordinate + clicked facade line
        ↓
Google Geocoding locates the building
        ↓
Overture Maps verifies footprint and building ID
        ↓
Clicked map line gives facade width in metres
        ↓
Facade mask gives pixel width and height
        ↓
height_m = width_m × (facade_height_px / facade_width_px)
        ↓
total_facade_area_m2 = width_m × height_m
        ↓
px_to_m2 = total_facade_area_m2 / facade_mask_pixel_count
        ↓
usable_area_m2 = usable_mask_pixel_count × px_to_m2
        ↓
panel_count = floor(usable_area_m2 / 1.7)
        ↓
capacity_kw = panel_count × 350 / 1000
        ↓
annual_kwh = capacity_kw × 950
```

This gives a transparent, verifiable, image-based BIPV facade estimation pipeline that can be explained, audited, and improved further with better facade segmentation or richer 3D building datasets.
