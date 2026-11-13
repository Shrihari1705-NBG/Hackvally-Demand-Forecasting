# Belagavi boundary validation

- geometry_checks_passed: `False`
- administrative_boundary_verified: `false`
- boundary_approved: `false`

## Official reference comparison

The City Corporation ward page and Cantonment Board pages were retained as comparison references. The supplied Cantonment city-map URL did not expose a downloadable boundary during this run. These references do not by themselves prove the OSM geometry is legally correct.

## Selected sources

### City Corporation

null

### Cantonment

null

## Merged coverage

{
  "geometry_type": null,
  "bounds": null,
  "area_km2": null,
  "component_count": null,
  "overlap_area_km2": null,
  "suspicious_findings": []
}

## Rejected candidates

[
  {
    "candidate": {
      "source": "Nominatim",
      "source_query": "Belagavi City Corporation, Karnataka, India",
      "retrieval_timestamp_utc": "2026-11-13T17:27:40.311620+00:00",
      "osm_element_type": "way",
      "osm_id": 511114422,
      "relation_id": null,
      "name": "Belagavi City Corporation",
      "official_name": null,
      "local_name": null,
      "display_name": "Belagavi City Corporation, Belgaum - Panaji Highway, Subhash Nagar, Belagavi, Belagavi taluku, Belagavi, Karnataka, 590001, India",
      "boundary": null,
      "admin_level": "Belagavi",
      "other_osm_tags": {},
      "geometry_type": "Polygon",
      "bounds": [
        74.5161952,
        15.8729083,
        74.516951,
        15.8736206
      ],
      "area_km2": null,
      "candidate_role": "city_corporation"
    },
    "reason": "No confidently identifiable administrative relation polygon"
  },
  {
    "candidate": {
      "source": "Nominatim",
      "source_query": "Belagavi Cantonment, Karnataka, India",
      "retrieval_timestamp_utc": "2026-11-13T17:27:40.311620+00:00",
      "osm_element_type": "way",
      "osm_id": 1119941070,
      "relation_id": null,
      "name": "Belagavi Cantonment",
      "official_name": null,
      "local_name": null,
      "display_name": "Belagavi Cantonment, Belagavi, Belagavi taluku, Belagavi, Karnataka, 590001, India",
      "boundary": null,
      "admin_level": "Belagavi",
      "other_osm_tags": {},
      "geometry_type": "Polygon",
      "bounds": [
        74.4764042,
        15.8438457,
        74.507904,
        15.8799668
      ],
      "area_km2": null,
      "candidate_role": "cantonment"
    },
    "reason": "No confidently identifiable administrative relation polygon"
  },
  {
    "candidate_role": "city_corporation",
    "reason": "No confidently identifiable administrative relation polygon"
  },
  {
    "candidate_role": "cantonment",
    "reason": "No confidently identifiable administrative relation polygon"
  }
]

## Manual verification required

Confirm that the selected City Corporation geometry covers the full corporation rather than only a locality, and that the Cantonment geometry includes both Main Cantonment and Fort Zone. Do not set `boundary_approved` automatically.

## Phase 2.5 alternative-source research

Decision: **C — neither boundary confidently identified**.

The ArcGIS `India_10_Cities_City_boundary` layer was queried directly. It contains Delhi, Bengaluru, Ahmedabad, Chennai, Hyderabad, Jaipur, Kolkata, Mumbai, Pune, and Surat; it does not contain Belagavi.

India Geodata documents municipal wards for 28 cities, sourced from SBM, AMRUT, and DataMeet. Its published urban file list did not include a Belagavi directory or ward file. Its district and taluka datasets are not acceptable substitutes for a City Corporation boundary.

The Karnataka State GIS page was investigated, but the page timed out in the browser fetch and failed certificate-chain verification in the direct read-only request. No downloadable Belagavi municipal or ward geometry could be confirmed.

The City Corporation ward-information page is accessible, but its exposed HTML contains only a dynamic ward-information interface; no downloadable ward polygons or GIS service were found. The Cantonment city-map URL redirected to the Board homepage and no downloadable boundary was exposed.

The OSM Cantonment candidate `way/1119941070` has `landuse=military` and an area of approximately 7.1574 km². This is close to the official 1,763.79-acre reference (~7.1378 km²), but the area agreement does not establish that it is the full legal Cantonment or that it includes both Main Cantonment and Fort Zone. It remains Low confidence and is not promoted.

See [boundary_candidate_comparison.csv](boundary_candidate_comparison.csv) and [boundary_source_inventory.json](boundary_source_inventory.json) for source, licensing, vintage, limitations, and suitability details.

## Required official input

Request from the City Corporation a current ward polygon dataset or official boundary in GeoJSON, Shapefile, GeoPackage, KML, or a georeferenced administrative map. Request from the Cantonment Board the official boundary geometry covering both Main Cantonment and Fort Zone in one of those GIS formats. Screenshots and textual locality descriptions are not sufficient.
