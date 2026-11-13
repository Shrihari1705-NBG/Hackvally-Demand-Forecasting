# Hourly H3 Demand Validation

**SYNTHETIC DEMAND DATA – NOT REAL-WORLD OBSERVATIONS**

- Rows: `406,080` / expected `406,080`
- Unique H3 cells: `94`
- Hourly intervals: `4320`
- Total demand: `137,291`
- Valid source rides: `137,291`
- Conservation passed: `True`
- Zero-demand cell-hours: `293,053`
- Non-zero cell-hours: `113,027`
- Demand min/mean/median/max: `0` / `0.3381` / `0.0` / `7`
- Timezone: `Asia/Kolkata`
- POI context status: `unavailable`

## Leakage protection

The model table does not copy ride IDs, coordinates, destinations, outcomes, validity flags, passenger identifiers, or lag/rolling features.
Zero POI counts, when present, do not prove absence of facilities.

## Demand by region

- `R01`: 22,411
- `R02`: 7,904
- `R03`: 7,546
- `R04`: 10,250
- `R05`: 17,888
- `R06`: 29,070
- `R07`: 18,833
- `R08`: 23,389

## Demand by hour

- `00:00`: 2,587
- `01:00`: 2,651
- `02:00`: 2,624
- `03:00`: 2,625
- `04:00`: 2,710
- `05:00`: 2,615
- `06:00`: 4,750
- `07:00`: 7,759
- `08:00`: 8,004
- `09:00`: 8,001
- `10:00`: 7,794
- `11:00`: 6,078
- `12:00`: 6,136
- `13:00`: 6,168
- `14:00`: 5,978
- `15:00`: 6,055
- `16:00`: 6,241
- `17:00`: 7,772
- `18:00`: 7,837
- `19:00`: 8,007
- `20:00`: 7,668
- `21:00`: 7,773
- `22:00`: 4,701
- `23:00`: 4,757

## Demand by weekday

- `0`: 20,673
- `1`: 20,834
- `2`: 20,317
- `3`: 21,022
- `4`: 20,052
- `5`: 17,132
- `6`: 17,261
