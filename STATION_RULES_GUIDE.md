# 🚂 Station Rules & Hand Signal Timing System Guide

## Overview

The system tracks hand signal compliance at specific stations with precise timing requirements. This guide explains how station rules work and how to configure them.

## 📁 Excel File Structure

### Required Columns

The Excel file (`Detected_Signals_Lat_Long_Enhanced.xlsx`) should have these columns:

1. **STATION_ID** (int) - Unique identifier for each station
2. **STATION_NAME** (string) - Human-readable station name
3. **LATITUDE** (float) - GPS latitude coordinate
4. **LONGITUDE** (float) - GPS longitude coordinate
5. **EXPECTED_TIME_SECONDS** (float) - **When the signal should be raised** (in video seconds)
6. **EXPECTED_TIME_FORMATTED** (string) - Human-readable time (HH:MM:SS)
7. **SIGNAL_REQUIRED** (bool) - Whether a signal is mandatory at this station
8. **TOLERANCE_SECONDS** (float) - Time tolerance (±seconds)
9. **TOLERANCE_METERS** (float) - GPS distance tolerance (meters)

### Example Excel Structure

```
STATION_ID | STATION_NAME    | LATITUDE  | LONGITUDE | EXPECTED_TIME_SECONDS | EXPECTED_TIME_FORMATTED | SIGNAL_REQUIRED | TOLERANCE_SECONDS | TOLERANCE_METERS
-----------|-----------------|-----------|-----------|----------------------|------------------------|-----------------|-------------------|------------------
1          | Central Station | 15.267390 | 73.980355 | 0                    | 00:00:00               | True            | 10                | 30
2          | North Junction | 15.267407 | 73.980444 | 60                   | 00:01:00               | True            | 10                | 30
3          | East Terminal  | 15.267404 | 73.980442 | 120                  | 00:02:00               | True            | 10                | 30
```

## ⏰ How Timing Works

### Expected Time

- **EXPECTED_TIME_SECONDS**: The exact video timestamp (in seconds) when the pilot should raise their hand
- Example: If `EXPECTED_TIME_SECONDS = 60`, the signal should be raised at 1 minute into the video

### Tolerance

- **TOLERANCE_SECONDS**: How much time deviation is allowed
- Example: If `EXPECTED_TIME_SECONDS = 60` and `TOLERANCE_SECONDS = 10`:
  - ✅ Signal raised between 50-70 seconds = **COMPLIANT**
  - ❌ Signal raised before 50 seconds = **EARLY**
  - ❌ Signal raised after 70 seconds = **LATE**
  - ❌ No signal detected = **MISSED**

## 📍 GPS Location Validation

### Distance Tolerance

- **TOLERANCE_METERS**: Maximum distance from station location (in meters)
- The system checks if the pilot's GPS location is within this distance when raising the signal
- Example: If `TOLERANCE_METERS = 30`:
  - ✅ Signal raised within 30m of station = **COMPLIANT**
  - ❌ Signal raised >30m from station = **VIOLATION** (wrong location)

## 🎯 Compliance Status

The system determines compliance based on:

1. **Time Accuracy**: Was the signal raised at the expected time (±tolerance)?
2. **Location Accuracy**: Was the signal raised at the correct GPS location?
3. **Signal Detection**: Was a hand signal actually detected?

### Status Types

- **COMPLIANT** ✅: Signal raised at correct time and location
- **MISSED** ❌: No signal detected at expected time
- **LATE** ⏰: Signal raised after tolerance window
- **EARLY** ⚡: Signal raised before tolerance window
- **VIOLATION** ⚠️: Signal raised at wrong location or poor quality

## 🔧 How to Create/Update Station Rules

### Option 1: Use the Enhancement Script

```bash
python3 enhance_station_rules.py
```

This will:
- Read `Detected_Signals_Lat_Long.xlsx`
- Add timing columns (default: every 60 seconds)
- Create `Detected_Signals_Lat_Long_Enhanced.xlsx`

### Option 2: Manual Excel Editing

1. Open `Detected_Signals_Lat_Long_Enhanced.xlsx`
2. Update `EXPECTED_TIME_SECONDS` with actual video timestamps
3. Update `STATION_NAME` with real station names
4. Adjust `TOLERANCE_SECONDS` if needed (default: 10 seconds)
5. Save the file

### Option 3: Use Real Railway Data

If you have MRT 922 data file:
```bash
python3 medha_data_parser.py
```

This creates station rules from actual locomotive data with real timing.

## 📊 Current Implementation

### Files Using Station Rules

1. **video_processor_api.py** - Main API that processes videos
2. **station_alert_system.py** - Validates station compliance
3. **perfect_hand_signal_detector.py** - Detects hand signals

### Code Flow

1. **Load Rules**: System loads Excel file on startup
2. **Process Video**: For each frame, check if near a station
3. **Detect Signal**: MediaPipe detects hand signals
4. **Validate**: Check time, location, and signal quality
5. **Report**: Generate compliance report with all stations

## 🎬 Example: How It Works in Practice

### Scenario

- **Station**: Central Station
- **Expected Time**: 60 seconds (00:01:00)
- **Tolerance**: ±10 seconds
- **GPS Location**: (15.267390, 73.980355)
- **Distance Tolerance**: 30 meters

### Video Processing

1. **At 00:00:55** - Pilot is 25m from Central Station
   - ✅ GPS location: Within tolerance
   - ⏰ Time: 5 seconds before expected (within tolerance)
   - ✅ Signal detected: Hand raised
   - **Result**: **COMPLIANT** ✅

2. **At 00:01:15** - Pilot is 35m from Central Station
   - ❌ GPS location: Outside tolerance (35m > 30m)
   - ⏰ Time: 15 seconds after expected (outside tolerance)
   - ✅ Signal detected: Hand raised
   - **Result**: **VIOLATION** ⚠️ (Wrong location + Late)

3. **At 00:01:00** - Pilot is at Central Station
   - ✅ GPS location: Within tolerance
   - ✅ Time: Exactly at expected time
   - ❌ Signal detected: No hand raised
   - **Result**: **MISSED** ❌

## 💡 Tips for Best Results

1. **Accurate Timing**: Use actual video timestamps, not estimated times
2. **GPS Precision**: Ensure GPS coordinates are accurate (6+ decimal places)
3. **Reasonable Tolerance**: 
   - Time: 10-15 seconds is usually good
   - Distance: 30-50 meters for railway stations
4. **Station Names**: Use descriptive names for easy identification
5. **Test First**: Process a short test video to verify timing

## 🔍 Troubleshooting

### Issue: "Using BASIC Excel format - generating timing rules"

**Solution**: The Excel file doesn't have `EXPECTED_TIME_SECONDS` column. Run:
```bash
python3 enhance_station_rules.py
```

### Issue: All signals marked as MISSED

**Possible Causes**:
- Timing in Excel doesn't match video timestamps
- GPS location not set correctly
- Hand signal detection confidence too low

**Solution**: 
- Check `EXPECTED_TIME_SECONDS` matches actual video times
- Verify GPS coordinates are correct
- Review hand signal detection logs

### Issue: Wrong station names

**Solution**: Update `STATION_NAME` column in Excel file with correct names

## 📝 Summary

- **Excel File**: `Detected_Signals_Lat_Long_Enhanced.xlsx` (or create with script)
- **Timing Column**: `EXPECTED_TIME_SECONDS` (when signal should be raised)
- **Tolerance**: `TOLERANCE_SECONDS` (how much time deviation allowed)
- **GPS Validation**: `TOLERANCE_METERS` (distance from station)
- **Status**: COMPLIANT, MISSED, LATE, EARLY, or VIOLATION

The system automatically uses the enhanced file if available, otherwise falls back to basic format with auto-generated timing.
