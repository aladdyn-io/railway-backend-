# Dual Excel Upload – Station Alerts with GPS Correlation

## Overview

You can now upload **two Excel files** on the Upload page to validate pilot hand signals using **actual train times** and **signal locations**:

1. **Signal Coordinates** – Lat/lng of each signal (e.g. `signal coordinates GDR-MAS UP.xlsx`)
2. **GPS Log (GDR TO MAS)** – Train journey GPS with timestamps (e.g. `GDR TO MAS.xlsx`)

Together, the system determines **when the train passed each signal** and checks whether the pilot raised their hand correctly or missed it at that time.

---

## How It Works

### 1. Signal Coordinates
- **File**: e.g. `signal coordinates GDR-MAS UP.xlsx`
- **Columns**: LATITUDE_RC, LONGITUDE_RC, SIG_CODE, STATION
- **Purpose**: Defines each signal location (174 points GDR→MAS)

### 2. GPS Log (GDR TO MAS)
- **File**: e.g. `GDR TO MAS.xlsx`
- **Columns**: Logging Time, Latitude, Longitude, Speed
- **Purpose**: Records the train’s position over time (7,299+ rows)

### 3. Correlation Logic
- For each signal point (lat, lng), the system finds the GPS log entry where the train was **nearest** to that location.
- The corresponding **Logging Time** becomes the expected time for that signal.
- Times are converted to “seconds from journey start.”
- During video processing, each detection is checked: “Was a hand signal detected within the expected time window?”

### 4. Validation Result
- **Pilot Did It ✓** – Hand raised in the correct time window  
- **Pilot Missed It ✗** – No hand raise in the required window  

---

## Usage

1. Open **Upload** in the app.
2. In **Station Alerts Excel Files**:
   - Upload **Signal Coordinates** (signal locations).
   - Upload **GPS Log** (train journey).
3. Upload your video.
4. Click **Upload & Process Video**.

### If You Upload Only Signal Coordinates
- Times are estimated by splitting the video duration across the signal points.
- Uploading both files gives more accurate timing.

### If You Upload Neither
- The backend uses disk files (`signal coordinates GDR-MAS UP.xlsx`) if present.
- Otherwise, station alerts validation is skipped.

---

## Files Changed

### Backend
- **requirement_team_loader.py**
  - `load_gps_log()` – Load GPS log from Excel (file or bytes)
  - `load_signal_rules_from_bytes()` – Load signal coords from uploaded Excel
  - `correlate_gps_with_signals()` – Map signal locations to train times

- **video_processor_api.py**
  - Accepts optional `signalCoords` and `gpsLog` in the form
  - Uses GPS correlation when both files are present

### Frontend
- **StationAlertsExcelUpload.tsx** (new) – Two upload areas for Signal Coords and GPS Log
- **UploadPage.tsx** – Integrates the component and sends both files to the API
- **api.ts** – `processVideo()` extended to accept optional Excel files
- **useVideoProcessing.ts** – `process()` accepts optional Excel files

---

## Expected File Formats

### Signal Coordinates
| Column       | Example    |
|-------------|------------|
| LATITUDE_RC | 14.144435  |
| LONGITUDE_RC| 79.844527  |
| SIG_CODE    | GDR STR:81 |
| STATION     | GDR        |

### GPS Log
| Column        | Example              |
|---------------|----------------------|
| Logging Time  | 2026-01-30 10:59:21  |
| Latitude      | 14.163380            |
| Longitude     | 79.849246            |
| Speed         | 54.46                |

---

## Summary

With both Excel files uploaded, the system uses real train timing and signal locations to validate whether the pilot raised their hand at the correct place and time.
