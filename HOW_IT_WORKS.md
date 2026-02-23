# ✅ How Signal Compliance Detection Works

## Overview

When you upload a video, the system will automatically:
1. ✅ Detect hand signals in the video
2. ✅ Match signals to station rules from Excel
3. ✅ Validate timing (when signal was raised vs expected time)
4. ✅ Validate location (GPS coordinates if available)
5. ✅ Generate compliance report

## 🔄 Complete Flow

### Step 1: Video Upload
- User uploads video through the frontend
- Video is sent to `/api/process-video` endpoint

### Step 2: Video Processing
- System processes video frame by frame (every 30th frame)
- For each frame:
  - Detects hand signals using MediaPipe
  - Records timestamp (video time in seconds)
  - Checks GPS location (if available)

### Step 3: Signal Detection
- MediaPipe analyzes each frame for hand signals
- Detects if hand is raised above shoulder/head
- Calculates confidence score (0-100%)
- Records detection timestamp

### Step 4: Rule Matching
- System loads station rules from Excel file:
  - `EXPECTED_TIME_SECONDS` - When signal should be raised
  - `STATION_NAME` - Station name
  - `TOLERANCE_SECONDS` - Time tolerance (±10 seconds default)
  - `LATITUDE/LONGITUDE` - Station location
  - `TOLERANCE_METERS` - Distance tolerance (30m default)

### Step 5: Compliance Validation

For each station rule, the system checks:

#### A. Timing Validation
- **Expected Time**: From `EXPECTED_TIME_SECONDS` in Excel
- **Actual Time**: When signal was detected in video
- **Tolerance**: From `TOLERANCE_SECONDS` in Excel

**Results:**
- ✅ **COMPLIANT**: Signal raised within tolerance (e.g., expected at 60s, raised at 55-65s)
- ⏰ **LATE**: Signal raised after tolerance window
- ⚡ **EARLY**: Signal raised before tolerance window
- ❌ **MISSED**: No signal detected at expected time

#### B. Location Validation (if GPS available)
- **Expected Location**: From `LATITUDE/LONGITUDE` in Excel
- **Actual Location**: GPS coordinates when signal was raised
- **Tolerance**: From `TOLERANCE_METERS` in Excel

**Results:**
- ✅ **COMPLIANT**: Signal raised within distance tolerance
- ⚠️ **VIOLATION**: Signal raised too far from station

### Step 6: Report Generation

The system generates a comprehensive report with:
- Total stations checked
- Number of compliant signals
- Number of missed signals
- Number of late/early signals
- Compliance rate percentage
- Detailed station-by-station breakdown

## 📊 Example Scenario

### Excel Rules (Detected_Signals_Lat_Long_Enhanced.xlsx)

| Station | Expected Time | Tolerance |
|---------|---------------|-----------|
| Station_001 | 0 seconds (00:00:00) | ±10 seconds |
| Station_002 | 60 seconds (00:01:00) | ±10 seconds |
| Station_003 | 120 seconds (00:02:00) | ±10 seconds |

### Video Processing

**Frame at 00:00:05 (5 seconds)**
- ✅ Hand signal detected
- ✅ Within tolerance for Station_001 (expected 0s, tolerance ±10s)
- **Result**: ✅ **COMPLIANT**

**Frame at 00:01:15 (75 seconds)**
- ✅ Hand signal detected
- ⏰ Late for Station_002 (expected 60s, detected at 75s = 15s late)
- **Result**: ⏰ **LATE** (outside ±10s tolerance)

**Frame at 00:02:00 (120 seconds)**
- ❌ No hand signal detected
- **Result**: ❌ **MISSED** (expected signal but none detected)

## 🎯 What You'll See in the UI

### 1. Summary Cards
- **Compliant Signals**: Number of correctly timed signals
- **Missed Signals**: Number of missing signals
- **Compliance Rate**: Percentage of stations with correct signals

### 2. Station Alert Compliance Table
Shows for each station:
- Station name
- Expected time (from Excel)
- Actual time (when signal was detected)
- Status (COMPLIANT, MISSED, LATE, EARLY)
- Time difference
- Compliance details

### 3. Event Timeline
- All hand signal events with timestamps
- Color-coded by status:
  - 🟢 Green: Compliant
  - 🔴 Red: Missed
  - 🟡 Yellow: Late/Early

## ⚙️ Configuration

### Excel File Structure
The system uses `Detected_Signals_Lat_Long_Enhanced.xlsx` with:
- `EXPECTED_TIME_SECONDS`: When signal should be raised
- `TOLERANCE_SECONDS`: How much time deviation is allowed
- `STATION_NAME`: Station identifier
- `SIGNAL_REQUIRED`: Whether signal is mandatory

### Customizing Rules
1. Open `Detected_Signals_Lat_Long_Enhanced.xlsx`
2. Update `EXPECTED_TIME_SECONDS` with actual video timestamps
3. Adjust `TOLERANCE_SECONDS` if needed (default: 10 seconds)
4. Update `STATION_NAME` with real station names
5. Save and restart the API

## 🔍 Troubleshooting

### Q: All signals marked as MISSED
**A**: Check that `EXPECTED_TIME_SECONDS` in Excel matches actual video timestamps

### Q: Signals marked as LATE/EARLY
**A**: Adjust `TOLERANCE_SECONDS` in Excel to allow more time deviation

### Q: Wrong station names
**A**: Update `STATION_NAME` column in Excel file

### Q: GPS validation not working
**A**: GPS is optional. Timing validation works without GPS. To enable GPS:
- Set GPS coordinates via `/api/set-gps` endpoint
- Or integrate GPS data from video metadata

## ✅ Summary

**YES, the system will recognize if the pilot shows correct signals according to rules!**

It validates:
- ✅ **Timing**: Was signal raised at the expected time?
- ✅ **Presence**: Was a signal detected at all?
- ✅ **Location**: Was signal raised at correct location? (if GPS available)

All validation is based on the rules in your Excel file (`Detected_Signals_Lat_Long_Enhanced.xlsx`).
