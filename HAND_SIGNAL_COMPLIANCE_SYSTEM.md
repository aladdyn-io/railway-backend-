# 🚂 Hand Signal Compliance System - Core Functionality

## Overview

This is the **MAIN PART** of the project: **Station Detection = Hand Signal Detection**

The system validates that hand signals are raised:
- ✅ **At the correct LOCATION** (GPS coordinates match station location within tolerance)
- ✅ **At the correct TIME** (signal raised at expected time within tolerance)

## Compliance Rules

### ✅ GREEN (COMPLIANT) - Signal is CORRECT
A hand signal is marked as **COMPLIANT** (GREEN) when:
1. ✅ Signal is detected (hand raised)
2. ✅ Location is correct (within tolerance distance from station)
3. ✅ Timing is correct (within tolerance time from expected time)
4. ✅ Confidence is good (≥ 50%)

### ❌ RED (VIOLATION) - Signal is WRONG
A hand signal is marked as **VIOLATION** (RED) when:
1. ❌ **MISSED SIGNAL**: At correct location but no signal detected
2. ❌ **TIMING ERROR**: Signal detected at correct location but wrong time (too early or too late)
3. ❌ **WRONG LOCATION**: Signal detected but not at the correct station
4. ❌ **POOR QUALITY**: Signal detected but confidence is too low (< 50%)

## How It Works

### 1. Excel File Configuration
The system loads station rules from `Detected_Signals_Lat_Long_Enhanced.xlsx` with:
- **STATION_NAME**: Name of the station
- **LATITUDE/LONGITUDE**: GPS coordinates of the station
- **EXPECTED_TIME_SECONDS**: Expected time in video when signal should be raised
- **TOLERANCE_SECONDS**: Time tolerance (default: ±10 seconds)
- **TOLERANCE_METERS**: Location tolerance (default: 30 meters)
- **SIGNAL_REQUIRED**: Whether signal is mandatory at this station

### 2. Detection Process

#### Step 1: Hand Signal Detection
- MediaPipe detects if hand is raised in the video frame
- Checks hand position relative to shoulders
- Validates finger extension
- Calculates confidence score (0-100%)

#### Step 2: Location Validation
- Gets current GPS coordinates from video metadata
- Finds nearest station using geopy distance calculation
- Checks if within tolerance distance (default: 30 meters)

#### Step 3: Timing Validation
- Gets current video timestamp
- Compares with expected time from Excel
- Checks if within tolerance time (default: ±10 seconds)

#### Step 4: Compliance Decision
- **COMPLIANT**: All checks pass (location ✅, timing ✅, signal ✅, confidence ✅)
- **VIOLATION**: Any check fails (location ❌, timing ❌, signal ❌, or confidence ❌)

### 3. UI Display

#### Green Indicators (COMPLIANT)
- ✅ Green badge: `bg-green-100 text-green-800 border-green-300`
- ✅ Green checkmark icon
- ✅ Status: "compliant"

#### Red Indicators (VIOLATION)
- ❌ Red badge: `bg-red-100 text-red-800 border-red-300`
- ❌ Red X icon or warning icon
- ❌ Status: "violation" or "missed"

## Code Structure

### Backend Files

1. **`perfect_hand_signal_detector.py`**
   - Hand signal detection using MediaPipe
   - Location validation (GPS distance)
   - Timing validation (expected time)
   - Compliance status determination

2. **`station_alert_system.py`**
   - Station rules loading from Excel
   - Station compliance tracking
   - Missed signal detection

3. **`video_processor_api.py`**
   - Main video processing
   - Integrates hand signal detection
   - Generates compliance reports

### Frontend Files

1. **`VideoAnalyticsPage.tsx`**
   - Displays compliance status
   - Green/Red color coding
   - Event timeline with status indicators

## Validation Logic

```python
# Location Check
within_location_tolerance = distance <= nearest_rule.tolerance_meters

# Timing Check
if expected_time is not None:
    time_diff = abs(timestamp_seconds - expected_time)
    timing_compliant = time_diff <= tolerance_seconds

# Compliance Decision
if (within_location_tolerance and 
    signal_detected and 
    timing_compliant and 
    confidence >= 50.0):
    status = "COMPLIANT"  # ✅ GREEN
else:
    status = "VIOLATION"  # ❌ RED
```

## Excel File Format

Required columns in `Detected_Signals_Lat_Long_Enhanced.xlsx`:

| Column | Description | Example |
|--------|-------------|---------|
| STATION_ID | Unique station identifier | 1, 2, 3... |
| STATION_NAME | Name of the station | "Central Station" |
| LATITUDE | GPS latitude | 40.7128 |
| LONGITUDE | GPS longitude | -74.0060 |
| EXPECTED_TIME_SECONDS | Expected time in video (seconds) | 120.5 |
| TOLERANCE_SECONDS | Time tolerance (±seconds) | 10.0 |
| TOLERANCE_METERS | Location tolerance (meters) | 30.0 |
| SIGNAL_REQUIRED | Whether signal is mandatory | TRUE/FALSE |

## Testing

To test the system:

1. **Upload a video** with GPS metadata
2. **Ensure Excel file** has station rules with timing
3. **Check console logs** for detection messages:
   - `✅ HAND SIGNAL DETECTED - COMPLIANT` (Green)
   - `❌ HAND SIGNAL DETECTED - VIOLATION` (Red)
4. **Review analytics page**:
   - Green badges for compliant signals
   - Red badges for violations
   - Detailed violation reasons

## Key Points

- **Location AND Timing** must BOTH be correct for compliance
- **Green = COMPLIANT** = Signal raised correctly at right place and time
- **Red = VIOLATION** = Signal missing, wrong location, wrong time, or poor quality
- This is the **CORE FUNCTIONALITY** of the project

## Status Colors

- 🟢 **GREEN** = `status: 'compliant'` = All checks passed
- 🔴 **RED** = `status: 'violation'` or `status: 'missed'` = Any check failed
