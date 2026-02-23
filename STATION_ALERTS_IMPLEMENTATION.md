# Station Alerts Implementation – Requirement Team Excel

## What Was Implemented

The system now uses **signal coordinates GDR-MAS UP.xlsx** from your requirement team to validate whether the pilot performed station alerts (hand raise) at each location.

### Flow

1. **Backend loads GDR-MAS rules**  
   `signal coordinates GDR-MAS UP.xlsx` (sheet `GDR-MAS`) is read automatically when processing a video.

2. **Time-window validation**  
   - 174 station/signal points are spread evenly across the video duration.  
   - Each point gets a time window (e.g. 0–20 s, 20–40 s, …).  
   - For each window: if a hand signal is detected → **Pilot Did It ✓**, otherwise → **Pilot Missed It ✗**.

3. **Frontend display**  
   The Analyze page shows:
   - **Pilot Did It ✓** – hand raise detected in the correct time window  
   - **Pilot Missed It ✗** – no hand raise in the required time window  

---

## Files Changed

### New
- **requirement_team_loader.py** – Loads GDR-MAS Excel and runs time-window validation

### Modified
- **video_processor_api.py** – Calls the loader and adds `signalValidation` to the API response
- **UploadPage.tsx** – Uses API `signalValidation` when present
- **VideoAnalyticsPage.tsx** – Uses API validation fields (stationName, location, etc.)
- **handSignalRules.ts** – Adds `loadRulesFromApiFormat()` for API rules

---

## File Location

Excel file must be in the project root:

```
/Users/yakesh/Desktop/Railways Project/signal coordinates GDR-MAS UP.xlsx
```

The loader checks the script directory and the current working directory.

---

## How to Test

1. Start the API:
   ```bash
   cd "/Users/yakesh/Desktop/Railways Project"
   python video_processor_api.py
   ```

2. Open the frontend, upload a video.

3. After processing, open the analytics page.

4. Check the **Station Alerts Compliance** section – it should show **Pilot Did It ✓** or **Pilot Missed It ✗** per location.

---

## Excel Format Used

| Column        | Example     | Purpose           |
|---------------|-------------|-------------------|
| LATITUDE_RC   | 14.144435   | Decimal degrees   |
| LONGITUDE_RC  | 79.844527   | Decimal degrees   |
| SIG_CODE      | GDR STR:81  | Signal/location   |
| STATION       | GDR         | Station code      |
| ROUTE_ID      | BZA-GDR-MAS | Route identifier  |

---

## Note on Time Windows

Without GPS in the video, time windows are estimated by splitting the video duration evenly across the 174 points. For Train 12711 (GDR 11:04 → MAS 12:58, ~114 min ≈ 6840 s), each point gets about 39 seconds. A hand signal detected in that window counts as **Pilot Did It ✓**.
