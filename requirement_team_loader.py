"""
Load station alert rules from Requirement Team Excel format.
Supports (uploaded or file):
  - Signal coordinates: LATITUDE, LONGITUDI (or LONGITUDE/LONGITUDE_RC), SIG_CODE, STATION, ROUTE_ID
  - Also: LATITUDE_RC, LONGITUDE_RC, SIG_CODE, STATION (GDR-MAS file)
GDR TO MAS.xlsx - GPS log with Logging Time, Latitude, Longitude - used to correlate signal times
"""
import os
import io
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from geopy.distance import geodesic


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and normalize column names for matching."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_gdr_mas_rules(excel_path: Optional[str] = None, 
                       sheet_name: str = "GDR-MAS") -> List[Dict[str, Any]]:
    """
    Load rules from requirement team GDR-MAS Excel format.
    Returns list of rule dicts with: station_name, location, latitude, longitude, route_id
    """
    if excel_path is None:
        # Search in script directory and common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "signal coordinates GDR-MAS UP.xlsx"),
            "signal coordinates GDR-MAS UP.xlsx",
        ]
        for p in candidates:
            if os.path.exists(p):
                excel_path = p
                break
        else:
            return []
    
    if not os.path.exists(excel_path):
        return []
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        
        # Requirement team format: LATITUDE_RC, LONGITUDE_RC, SIG_CODE, STATION, ROUTE_ID
        required_cols = ['LATITUDE_RC', 'LONGITUDE_RC', 'SIG_CODE', 'STATION']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ Missing column {col} in {excel_path}")
                return []
        
        rules = []
        for idx, row in df.iterrows():
            loc = str(row['SIG_CODE']).strip() if pd.notna(row['SIG_CODE']) else ''
            rules.append({
                'sid': loc or f'SID_{idx+1}',
                'station_name': str(row['STATION']).strip(),
                'location': loc,
                'latitude': float(row['LATITUDE_RC']),
                'longitude': float(row['LONGITUDE_RC']),
                'route_id': str(row['ROUTE_ID']).strip() if 'ROUTE_ID' in df.columns and pd.notna(row.get('ROUTE_ID')) else 'BZA-GDR-MAS',
                'sequence': idx + 1,
                'required_signal': 'hand_raised'
            })
        
        print(f"✅ Loaded {len(rules)} station alert rules from {excel_path} (GDR-MAS)")
        return rules
        
    except Exception as e:
        print(f"❌ Error loading {excel_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

33
def load_gps_log(excel_path_or_bytes: Optional[Union[str, bytes]] = None,
                 sheet_name: Optional[Union[int, str]] = None) -> List[Dict[str, Any]]:
    """
    Load GPS log from GDR TO MAS.xlsx format.
    Columns: Logging Time (DD-MM-YYYY HH:MM:SS or DD/MM/YY HH:MM), Latitude, Longitude, Speed, Direction, Distance, Status, GDR.
    Parses time with dayfirst=True. Returns list of {timestamp_seconds, latitude, longitude, ...} sorted by time.
    """
    if excel_path_or_bytes is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "GDR TO MAS.xlsx")
        if not os.path.exists(path):
            return []
        excel_path_or_bytes = path

    # When loading from bytes (uploaded file), try sheet 0 then "GDR TO MAS"
    sheets_to_try: List[Union[int, str]] = [sheet_name] if sheet_name is not None else [0, "GDR TO MAS"]
    if isinstance(excel_path_or_bytes, bytes) and sheet_name is None:
        try:
            xl = pd.ExcelFile(io.BytesIO(excel_path_or_bytes))
            if xl.sheet_names:
                sheets_to_try = [0] + [n for n in xl.sheet_names if n not in (0,)]
        except Exception:
            sheets_to_try = [0, "GDR TO MAS"]

    for try_sheet in sheets_to_try:
        try:
            if isinstance(excel_path_or_bytes, bytes):
                df = pd.read_excel(io.BytesIO(excel_path_or_bytes), sheet_name=try_sheet, header=0)
            else:
                df = pd.read_excel(excel_path_or_bytes, sheet_name=try_sheet, header=0)
        except Exception:
            continue
        df = _normalize_columns(df)

        # GDR TO MAS format: Logging Time, Latitude, Longitude, Speed
        time_col = None
        for c in ['Logging Time', 'LoggingTime', 'Time', 'Timestamp', 'Date', 'DateTime']:
            for col in df.columns:
                if str(col).lower() == c.lower() or c.lower() in str(col).lower():
                    time_col = col
                    break
            if time_col is not None:
                break
        if time_col is None:
            time_col = df.columns[0]

        lat_col = next((c for c in df.columns if 'lat' in str(c).lower()), None)
        lng_col = next((c for c in df.columns if 'lon' in str(c).lower()), None)
        if lat_col is None or lng_col is None:
            continue

        df = df.dropna(subset=[time_col, lat_col, lng_col])
        if df.empty:
            continue
        df = df.sort_values(time_col).reset_index(drop=True)

        # Parse dates as DD/MM/YY HH:MM (dayfirst=True for European format)
        first_time = pd.to_datetime(df[time_col].iloc[0], dayfirst=True)
        entries = []
        speed_col = next((c for c in df.columns if 'speed' in str(c).lower()), None)
        for _, row in df.iterrows():
            t = pd.to_datetime(row[time_col], dayfirst=True)
            seconds_from_start = (t - first_time).total_seconds()
            sp = float(row[speed_col]) if speed_col and pd.notna(row.get(speed_col)) else 0.0
            # real_datetime for OCR matching: video clip time vs Excel time
            real_dt = first_time + pd.Timedelta(seconds=seconds_from_start)
            entries.append({
                'timestamp_seconds': seconds_from_start,
                'real_datetime': real_dt.to_pydatetime() if hasattr(real_dt, 'to_pydatetime') else real_dt,
                'latitude': float(row[lat_col]),
                'longitude': float(row[lng_col]),
                'speed': sp,
            })
        if entries:
            print(f"✅ Loaded {len(entries)} GPS log entries from sheet {try_sheet!r} (journey: {entries[-1]['timestamp_seconds']/60:.1f} min)")
            return entries

    print("⚠️ GPS log: no sheet had valid Logging Time, Latitude, Longitude. Check file format.")
    return []


def correlate_sid_with_rtis(signal_rules: List[Dict], rtis_entries: List[Dict], 
                            tolerance_seconds: float = 3.0, top_n: int = 5) -> List[Dict]:
    """
    User algorithm: For each SID (signal station) in the list:
    1. Get SID(latitude), SID(longitude)
    2. Find the 5 closest (lat, lng) points in RTIS report
    3. Average their times T1..T5 → expected time
    4. Use ±3 seconds from expected time for hand raise check
    Also sets expected_real_datetime for OCR-based matching (video on-screen time vs Excel).
    """
    from datetime import timedelta
    if not signal_rules or not rtis_entries:
        return signal_rules

    journey_end = rtis_entries[-1]['timestamp_seconds']
    first_real = rtis_entries[0].get('real_datetime')  # For OCR matching

    for rule in signal_rules:
        sid_lat = rule['latitude']
        sid_lng = rule['longitude']

        # Find distances to all RTIS points
        distances = []
        for gps in rtis_entries:
            dist_m = geodesic((sid_lat, sid_lng), (gps['latitude'], gps['longitude'])).meters
            distances.append((dist_m, gps['timestamp_seconds']))

        # Sort by distance, take top 5 closest
        distances.sort(key=lambda x: x[0])
        top_5 = distances[:min(top_n, len(distances))]

        if top_5:
            times = [t for _, t in top_5]
            expected_time = sum(times) / len(times)
            rule['expected_time_seconds'] = expected_time
            rule['time_window'] = {
                'start': max(0, expected_time - tolerance_seconds),
                'end': min(journey_end, expected_time + tolerance_seconds)
            }
            rule['tolerance_seconds'] = tolerance_seconds
            rule['_closest_distances_m'] = [d for d, _ in top_5]
            rule['_avg_from_n_points'] = len(top_5)
            if first_real:
                rule['expected_real_datetime'] = first_real + timedelta(seconds=expected_time)
        else:
            # Fallback if no RTIS data
            idx = rule.get('sequence', 0) - 1
            n = len(signal_rules)
            expected_time = (idx / max(1, n - 1)) * journey_end if n > 1 else journey_end / 2
            rule['expected_time_seconds'] = expected_time
            rule['time_window'] = {
                'start': max(0, expected_time - tolerance_seconds),
                'end': min(journey_end, expected_time + tolerance_seconds)
            }
            rule['tolerance_seconds'] = tolerance_seconds
            if first_real:
                rule['expected_real_datetime'] = first_real + timedelta(seconds=expected_time)

    print(f"✅ Correlated {len(signal_rules)} SIDs with RTIS (top {top_n} points, ±{tolerance_seconds}s)")
    return signal_rules


def correlate_gps_with_signals(signal_rules: List[Dict], gps_entries: List[Dict], 
                               tolerance_meters: float = 80.0) -> List[Dict]:
    """Legacy: use correlate_sid_with_rtis with ±3 seconds and top 5 points."""
    return correlate_sid_with_rtis(signal_rules, gps_entries, tolerance_seconds=3.0, top_n=5)


def _find_column(df: pd.DataFrame, candidates: List[str], contains: Optional[str] = None) -> Optional[str]:
    """Find first column that exists (case-insensitive), or that contains substring (case-insensitive)."""
    col_lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c and str(c).lower() in col_lower:
            return col_lower[str(c).lower()]
    if contains:
        for col in df.columns:
            if contains.lower() in str(col).lower():
                return col
    return None


def _try_parse_signal_sheet(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Try to parse a DataFrame as signal rules. Returns non-empty list on success."""
    df = _normalize_columns(df)
    # LATITUDE / LONGITUDE (file may have LONGITUDI if header was truncated)
    lat_col = _find_column(df, ['LATITUDE_RC', 'LATITUDE', 'Latitude', 'LAT', 'Lat'], 'lat')
    lng_col = _find_column(df, ['LONGITUDE_RC', 'LONGITUDE', 'LONGITUDI', 'Longitude', 'LON', 'Lng', 'Long'], 'lon')
    if not lat_col or not lng_col:
        return []
    sid_col = _find_column(df, ['SID', 'SIG_CODE', 'Signal', 'Location', 'SIGNAL'], 'sig') or _find_column(df, [], 'code')
    stn_col = _find_column(df, ['STATION', 'Station', 'STN'], 'station')
    route_col = _find_column(df, ['ROUTE_ID', 'Route', 'ROUTE'], 'route')
    sig_col = sid_col or (list(df.columns)[2] if len(df.columns) > 2 else None)
    stn_col = stn_col or (list(df.columns)[3] if len(df.columns) > 3 else None)
    rules = []
    for idx, row in df.iterrows():
        try:
            if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
                continue
            lat_val = float(row[lat_col])
            lng_val = float(row[lng_col])
        except (TypeError, ValueError):
            continue
        sid_val = str(row.get(sig_col, '')).strip() if sig_col and pd.notna(row.get(sig_col)) else ''
        rules.append({
            'sid': sid_val or f'SID_{len(rules)+1}',
            'station_name': str(row.get(stn_col, '')).strip() if stn_col and pd.notna(row.get(stn_col)) else (sid_val or f'Point_{len(rules)+1}'),
            'location': sid_val,
            'latitude': lat_val,
            'longitude': lng_val,
            'route_id': str(row.get(route_col, '')).strip() if route_col and pd.notna(row.get(route_col)) else 'BZA-GDR-MAS',
            'sequence': len(rules) + 1,
            'required_signal': 'hand_raised'
        })
    return rules


def load_signal_rules_from_bytes(data: bytes, sheet_name: Union[int, str, None] = None) -> List[Dict[str, Any]]:
    """Load signal coordinates from Excel bytes (uploaded file). Tries all sheets with flexible column names."""
    if not data or len(data) < 100:
        print("❌ Signal coordinates file empty or too small")
        return []
    try:
        xl = pd.ExcelFile(io.BytesIO(data))
    except Exception as e:
        print(f"❌ Could not open Excel: {e}")
        return []
    # Try sheet indices and common names first (SIGNAL CORDINATES GDR TO MAS UP file)
    to_try: List[Union[int, str]] = [0, 1, 2, "GDR-MAS", "GDR-MAS UP", "GDR TO MAS UP", "Sheet1", "Signals"]
    for name in xl.sheet_names:
        if name not in to_try:
            to_try.append(name)
    seen: set = set()
    for attempt in to_try:
        if attempt in seen:
            continue
        seen.add(attempt)
        try:
            df = pd.read_excel(io.BytesIO(data), sheet_name=attempt, header=0)
        except Exception:
            continue
        if df is None or df.empty or len(df.columns) < 2:
            continue
        rules = _try_parse_signal_sheet(df)
        if rules:
            print(f"✅ Loaded {len(rules)} signal rules from uploaded file (sheet={attempt})")
            return rules
    # Last resort: first sheet, use first two numeric columns as lat/lng
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name=0, header=0)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            lat_col, lng_col = numeric_cols[0], numeric_cols[1]
            rules = []
            for idx, row in df.iterrows():
                if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
                    continue
                rules.append({
                    'station_name': str(row.iloc[2]) if len(df.columns) > 2 else f'Point_{len(rules)+1}',
                    'location': str(row.iloc[3]) if len(df.columns) > 3 else '',
                    'latitude': float(row[lat_col]),
                    'longitude': float(row[lng_col]),
                    'route_id': 'BZA-GDR-MAS',
                    'sequence': len(rules) + 1,
                    'required_signal': 'hand_raised'
                })
            if rules:
                print(f"✅ Loaded {len(rules)} signal rules (numeric columns fallback)")
                return rules
    except Exception:
        pass
    print("❌ Could not parse signal coordinates Excel - need columns like Latitude/Longitude (or LATITUDE_RC/LONGITUDE_RC)")
    return []


def assign_time_windows(rules: List[Dict], video_duration_seconds: float, 
                       tolerance_seconds: float = 15.0) -> List[Dict]:
    """
    Assign expected time windows to rules based on video duration.
    Distributes stations evenly across the video timeline.
    """
    if not rules or video_duration_seconds <= 0:
        return rules
    
    n = len(rules)
    window_width = video_duration_seconds / n
    
    for i, rule in enumerate(rules):
        # Each station gets a time window - midpoint is expected time
        start = max(0, i * window_width - tolerance_seconds)
        end = min(video_duration_seconds, (i + 1) * window_width + tolerance_seconds)
        expected_time = (start + end) / 2
        
        rule['time_window'] = {'start': start, 'end': end}
        rule['expected_time_seconds'] = expected_time
        rule['tolerance_seconds'] = min(tolerance_seconds, window_width / 2)
    
    return rules


def _parse_ocr_timestamp(d: Dict) -> Optional[Any]:
    """Parse ocr_timestamp from detection (datetime or ISO string)."""
    v = d.get('ocr_timestamp')
    if v is None:
        return None
    if hasattr(v, 'timestamp'):  # datetime
        return v
    if isinstance(v, str):
        try:
            from datetime import datetime
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        except Exception:
            return None
    return None


def validate_station_alerts(rules: List[Dict], detection_results: List[Dict]) -> List[Dict]:
    """
    Validate detection results against station rules.
    Uses OCR on-screen time when available (video clip): match ocr_timestamp to expected_real_datetime.
    Otherwise uses video timeline (timestamp_seconds vs time_window).
    Returns: list of validation results for frontend
    """
    validation_results = []
    
    for rule in rules:
        rule_id = f"rule_{rule['sequence']}"
        time_window = rule.get('time_window', {})
        start = time_window.get('start', 0)
        end = time_window.get('end', 0)
        expected_real = rule.get('expected_real_datetime')
        tol = rule.get('tolerance_seconds', 3.0)
        
        # Prefer OCR matching when both rule and detections have real-time data
        matching = []
        if expected_real:
            exp_dt = getattr(expected_real, 'to_pydatetime', lambda: expected_real)()
            for d in detection_results:
                if not d.get('handSignal', {}).get('signal_detected'):
                    continue
                ocr_ts = _parse_ocr_timestamp(d)
                if ocr_ts is not None:
                    delta = abs((ocr_ts - exp_dt).total_seconds())
                    if delta <= tol:
                        matching.append((d, delta))
        
        if matching:
            # OCR matching: use closest match
            matching.sort(key=lambda x: x[1])
            best = matching[0][0]
        else:
            # Fallback: video timeline (timestamp_seconds vs time_window)
            timeline_matches = [
                d for d in detection_results
                if (d.get('handSignal', {}).get('signal_detected') == True and
                    start <= d.get('timestamp_seconds', 0) <= end)
            ]
            best = min(timeline_matches, key=lambda x: abs(x.get('timestamp_seconds', 0) - (start + end) / 2)) if timeline_matches else None
        
        if best:
            # Pilot DID the station alert - hand was raised in time window
            sig_type = best.get('handSignal', {}).get('signal_type', 'hand_raised')
            confidence = best.get('handSignal', {}).get('confidence', 0.8)
            if isinstance(confidence, (int, float)):
                pass
            else:
                confidence = 0.8
            
            validation_results.append({
                'ruleId': rule_id,
                'sid': rule.get('sid', rule.get('location', '')),
                'stationName': rule.get('station_name', ''),
                'stationAlertSignal': rule.get('location', '') or rule.get('sid', ''),
                'detected': True,
                'timestamp': best.get('timestamp_seconds', start),
                'signalType': sig_type or 'hand_raised',
                'compliance': 'compliant',
                'confidence': float(confidence) if confidence else 0.8,
                'location': rule.get('location', ''),
                'latitude': rule.get('latitude'),
                'longitude': rule.get('longitude'),
                'requiredSignal': rule.get('required_signal', 'hand_raised'),
                'expectedTimeSeconds': rule.get('expected_time_seconds'),
                'timeWindow': rule.get('time_window'),
            })
        else:
            # Pilot MISSED the station alert - no hand raise in ±3s window
            validation_results.append({
                'ruleId': rule_id,
                'sid': rule.get('sid', rule.get('location', '')),
                'stationName': rule.get('station_name', ''),
                'stationAlertSignal': rule.get('location', '') or rule.get('sid', ''),
                'detected': False,
                'timestamp': rule.get('expected_time_seconds', start),
                'signalType': 'none',
                'compliance': 'missed',
                'confidence': 0,
                'location': rule.get('location', ''),
                'latitude': rule.get('latitude'),
                'longitude': rule.get('longitude'),
                'requiredSignal': rule.get('required_signal', 'hand_raised'),
                'expectedTimeSeconds': rule.get('expected_time_seconds'),
                'timeWindow': rule.get('time_window'),
            })
    
    return validation_results


def get_rules_for_frontend(rules: List[Dict]) -> List[Dict]:
    """Convert rules to frontend HandSignalRule format"""
    return [
        {
            'id': f"rule_{r['sequence']}",
            'sid': r.get('sid', r.get('location', '')),
            'stationName': r.get('station_name', ''),
            'stationAlertSignal': r.get('location', '') or r.get('sid', ''),
            'location': r.get('location', ''),
            'requiredSignal': r.get('required_signal', 'hand_raised'),
            'timeWindow': r.get('time_window', {'start': 0, 'end': 0}),
            'coordinates': {'lat': r.get('latitude', 0), 'lng': r.get('longitude', 0)},
            'mandatory': True,
            'description': f"{r.get('station_name', '')} | {r.get('location', '')}"
        }
        for r in rules
    ]
