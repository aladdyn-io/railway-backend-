# 📍 GPS Data Requirements for Station Alert System

## Current Status

### ✅ What We Have:

1. **Station GPS Coordinates** (from Excel file):
   - `LATITUDE` - Station latitude
   - `LONGITUDE` - Station longitude
   - `TOLERANCE_METERS` - Distance tolerance (default: 30 meters)

2. **Station Timing** (from Excel file):
   - `EXPECTED_TIME_SECONDS` - When signal should be raised
   - `TOLERANCE_SECONDS` - Time tolerance (default: ±10 seconds)

3. **Station Information**:
   - `STATION_ID` - Unique identifier
   - `STATION_NAME` - Station name
   - `SIGNAL_REQUIRED` - Whether signal is mandatory

### ❓ What We Need:

**GPS Data from Video** - To compare current location with station locations

## How GPS Should Work:

### Option 1: GPS from Video Metadata
If your videos have GPS metadata embedded:
- Extract GPS coordinates from video file
- Use `set_current_gps(latitude, longitude)` to set current location
- System compares video GPS with station GPS

### Option 2: GPS from External Source
If GPS comes from external system:
- Provide GPS coordinates with video upload
- Pass GPS data in API request
- System uses provided GPS coordinates

### Option 3: Simulated GPS (for testing)
If no GPS available:
- System can work with timing only
- Location validation will be skipped
- Only timing validation will work

## Current Implementation:

The system currently expects GPS to be set via:
```python
processor.perfect_hand_detector.set_current_gps(latitude, longitude)
```

But this is **NOT automatically called** - GPS must be provided!

## Questions for You:

1. **Do your videos have GPS metadata?**
   - If yes, we need to extract it from video file
   - If no, we need GPS from another source

2. **How do you get GPS coordinates?**
   - Embedded in video file?
   - From external GPS device?
   - From route/map data?
   - Need to simulate for testing?

3. **What GPS format do you have?**
   - Decimal degrees (e.g., 15.26739, 73.980355)?
   - Degrees/minutes/seconds?
   - Other format?

## What We Need to Add:

1. **GPS Extraction from Video** (if available):
   ```python
   # Extract GPS from video metadata
   gps_data = extract_gps_from_video(video_path)
   if gps_data:
       processor.perfect_hand_detector.set_current_gps(
           gps_data['latitude'], 
           gps_data['longitude']
       )
   ```

2. **GPS from API Request** (if provided):
   ```python
   # Get GPS from request
   latitude = request.form.get('latitude')
   longitude = request.form.get('longitude')
   if latitude and longitude:
       processor.perfect_hand_detector.set_current_gps(
           float(latitude), 
           float(longitude)
       )
   ```

3. **GPS Update During Processing** (if GPS changes):
   ```python
   # Update GPS as video progresses (if GPS track available)
   for frame in video:
       current_gps = get_gps_at_timestamp(timestamp)
       processor.perfect_hand_detector.set_current_gps(
           current_gps['latitude'],
           current_gps['longitude']
       )
   ```

## Current Workaround:

If GPS is not available, the system will:
- ✅ Still detect hand signals
- ✅ Still validate timing
- ⚠️ Skip location validation
- ⚠️ Show "NO_GPS" status

## Next Steps:

**Please tell me:**
1. Do you have GPS data in your videos?
2. How is GPS data provided/stored?
3. What format is the GPS data in?

Then I can implement the GPS extraction/usage properly!
