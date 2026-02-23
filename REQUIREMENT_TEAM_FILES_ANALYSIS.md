# Requirement Team Files – Analysis & Integration Plan

## Your Context
- **Train No:** 12711  
- **Loco No:** 30239  
- **Route:** GUDUR (GDR) DEP 11:04 → MAS (Chennai) ARR 12:58  
- **LP Side CVVRS** – Locomotive Pilot side Camera-based Video Vigilance Recording System  

---

## Files Verified ✓

| File | Status | Rows | Purpose |
|------|--------|------|---------|
| **signal coordinates GDR-MAS UP.xlsx** | ✓ Found | 174 | **Primary – Station rules for GDR→MAS** |
| **GDR TO MAS.xlsx** | ✓ Found | 7,299 | GPS log data (actual trip timestamps) |
| **ROUTE19 MAS AJJ FL with details.xlsx** | ✓ Found | 81 | MAS→AJJ route (Fast Line) |
| ROUTE17 VLCY AJJ SL.CSV | ✓ Found | — | VLCY→AJJ Slow Line |
| ROUTE18 AJJ VLCY SL.CSV | ✓ Found | — | AJJ→VLCY Slow Line |
| ROUTE19 MAS AJJ FL.CSV | ✓ Found | — | MAS→AJJ Fast Line |
| ROUTE20 AJJ MAS F L.CSV | ✓ Found | — | AJJ→MAS Fast Line |

**Location:** `/Users/yakesh/Desktop/Railways Project/`

---

## signal coordinates GDR-MAS UP.xlsx (PRIMARY FOR TRAIN 12711)

**Sheet: GDR-MAS** | 174 station/signal points | Route: BZA-GDR-MAS

| Column | Example | Notes |
|--------|---------|------|
| LATITUDE_RC | 14.144435 | **Decimal degrees** ✓ (no conversion needed) |
| LONGITUDE_RC | 79.844527 | **Decimal degrees** ✓ |
| SIG_CODE | GDR STR:81, GDR L XING:99 | Signal/location code |
| STATION | GDR, ODR, MAS | Station code |
| ROUTE_ID | BZA-GDR-MAS | Route identifier |

**Sample rows:**
```
14.144435, 79.844527, GDR STR:81,    GDR
14.140063, 79.844078, GDR L XING:99, GDR
14.137583, 79.844767, GDR SA:78,     GDR
...
```

This file is **ideal** for Station Alerts – coordinates are already in decimal degrees.

---

## GDR TO MAS.xlsx (GPS LOG)

**7,299 rows** – Actual trip GPS data with timestamps.

| Column | Example |
|--------|---------|
| Device Id | 3014 |
| Logging Time | 2026-01-30 10:59:21 |
| Latitude | 14.163380 |
| Longitude | 79.849246 |
| Speed | 54.46 |
| last/cur stationCode | GDR |

**Use:** Map video timestamps to GPS position; estimate expected time at each signal point from this log.

---

## ROUTE19 MAS AJJ FL with details.xlsx

**81 rows** – MAS→AJJ Fast Line. Columns: Seq, Station, Event, Lat (DDMM.MMMM), Long, Signal type (STR), etc.

---

## CSV Structure (ROUTE files)

### Header row (line 1)
```
Route_ID, Route_Name, Count, Direction, (empty columns...)
```
Example: `19,MAS-AJJ-FAST,20,U,,,,,,,,,,`

### Data rows (line 2+)
| Col | Example | Meaning |
|-----|---------|---------|
| 1 | 1 | Sequence no. |
| 2 | MAS, BBQ, VPY | Station code |
| 3 | MAS STR, BBQ HOME:2A | Location / signal point |
| 4 | 500 | Constant |
| 5 | 20 | Constant |
| 6 | 1305.3054 | **Latitude** (DDMM.MMMM – degrees + decimal minutes) |
| 7 | 8016.4717 | **Longitude** (DDMM.MMMM) |
| 8 | S | Direction (S/N) |
| 9 | 333.44 | Likely speed/parameter |
| 10 | 11, 7, 13 | Signal/location type code |
| 11–13 | E, E, E | Flags |
| 14 | STR | Signal type (STR, etc.) |

### Coordinate format
- Example: `1305.3054, 8016.4717`
- Interpretation: **13°05.3054′ N, 80°16.4717′ E** (DDMM.MMMM)
- Conversion:  
  - Lat: 13 + 5.3054/60 ≈ 13.0884°  
  - Lng: 80 + 16.4717/60 ≈ 80.2745°

---

## Route Summary

| File | Route | Direction | Type |
|------|-------|-----------|------|
| ROUTE17 | VLCY → AJJ | Slow Line | Villivakkam to Arakkonam |
| ROUTE18 | AJJ → VLCY | Slow Line | Arakkonam to Villivakkam |
| ROUTE19 | MAS → AJJ | Fast Line | Chennai to Arakkonam |
| ROUTE20 | AJJ → MAS | Fast Line | Arakkonam to Chennai |

For **Train 12711 (GDR → MAS)** the most relevant are:
- **GDR TO MAS.xlsx** or **signal coordinates GDR-MAS UP.xlsx** (Gudur → Chennai)
- ROUTE19/20 may overlap with the Chennai–Arakkonam portion, depending on the actual path.

---

## Mapping to CVVRS Station Alerts

Current CVVRS expects:

| Our System Expects | From Requirement Team CSV |
|--------------------|---------------------------|
| Station name | Col 2 + Col 3 (e.g. "MAS - MAS STR") |
| Latitude | Col 6 (convert DDMM.MMMM → decimal degrees) |
| Longitude | Col 7 (convert DDMM.MMMM → decimal degrees) |
| Signal required | Col 14 (STR, SA, etc.) – treat as “hand raise / alert” |
| Expected time | Not in CSV – derive from video duration / route order |
| Location description | Col 3 (e.g. BBQ HOME:2A, BBQ SA:7) |

---

## Integration Plan

### 1. Converter script (CSV → system format)
- Read ROUTE CSV files (and Excel when available).
- Convert DDMM.MMMM to decimal degrees.
- Map columns to:
  - `station_name`, `location`, `latitude`, `longitude`, `signal_type`, `coordinates`.
- Optionally estimate `start_time` / `end_time` based on sequence and average spacing (e.g. every 60 s).

### 2. Route selection
- For GDR → MAS, use **GDR TO MAS.xlsx** or **signal coordinates GDR-MAS UP.xlsx** when you add them.
- If those are missing, use ROUTE19/20 as a temporary source for the Chennai–Arakkonam part, and extend when GDR–MAS files are available.

### 3. Frontend (RulesUpload)
- Add real Excel/CSV parsing (e.g. `xlsx` library).
- Support both:
  - Our existing format (Station Name, Location, Signal Type, Start Time, End Time, Lat, Long).
  - Requirement team format (after conversion).

### 4. Backend (Python API)
- Add a CSV/Excel parser for the requirement team format.
- Convert coordinates and map to `PerfectHandSignalDetector` / `StationAlertSystem` format.

---

## Next Steps

1. **Place files:** Put the Excel files here:
   ```
   /Users/yakesh/Desktop/Railways Project/
   ```
2. **Share format:** If possible, share:
   - Column layout of **signal coordinates GDR-MAS UP.xlsx**
   - Column layout of **GDR TO MAS.xlsx**
   - Sample rows from **ROUTE19/20 with details.xlsx**
3. **Implementation:** After you add the Excel files and confirm structure, we can:
   - Add a converter script for CSV/Excel → system format
   - Update RulesUpload to parse real files
   - Update backend to load from the requirement team format

---

## Summary

- 4 ROUTE CSV files are available and understood.
- Excel files (GDR–MAS, signal coordinates, ROUTE with details) are not in the project yet.
- CSV structure is clear; coordinate conversion and column mapping are defined.
- Once the Excel files are in place and we see their structure, we can wire everything into the Station Alerts (hand raise) system.
