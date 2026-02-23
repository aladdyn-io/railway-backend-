import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from geopy.distance import geodesic

@dataclass
class StationRule:
    """Station rule from Excel file"""
    station_id: int
    station_name: str
    latitude: float
    longitude: float
    expected_time: float  # Expected time in video (seconds)
    signal_required: bool
    tolerance_seconds: float = 10.0  # ±10 seconds tolerance
    tolerance_meters: float = 50.0   # ±50 meters tolerance

@dataclass
class StationAlert:
    """Station alert compliance result"""
    station_rule: StationRule
    expected_time: str
    actual_signal_time: Optional[str]
    status: str  # 'COMPLIANT', 'MISSED', 'LATE', 'EARLY'
    time_difference: Optional[float]
    distance_difference: Optional[float]
    compliance_details: str

class StationAlertSystem:
    def __init__(self, excel_file_path: str = "Detected_Signals_Lat_Long_Enhanced.xlsx"):
        self.excel_file_path = excel_file_path
        self.station_rules = self.load_station_rules()
        
    def load_station_rules(self) -> List[StationRule]:
        """Load station rules from enhanced Excel with timing data"""
        try:
            df = pd.read_excel(self.excel_file_path)
            rules = []
            
            # Check if enhanced format with timing data exists
            if 'EXPECTED_TIME_SECONDS' in df.columns:
                print("✅ Using ENHANCED Excel format with timing rules")
                for idx, row in df.iterrows():
                    rule = StationRule(
                        station_id=row.get('STATION_ID', idx),
                        station_name=row.get('STATION_NAME', f"Station_{idx+1:03d}"),
                        latitude=row['LATITUDE'],
                        longitude=row['LONGITUDE'],
                        expected_time=row['EXPECTED_TIME_SECONDS'],
                        signal_required=row.get('SIGNAL_REQUIRED', True),
                        tolerance_seconds=row.get('TOLERANCE_SECONDS', 10.0),
                        tolerance_meters=30.0
                    )
                    rules.append(rule)
            else:
                print("⚠️ Using BASIC Excel format - generating timing rules")
                # Fallback to old method for basic Excel files
                for idx, row in df.iterrows():
                    expected_time = idx * 60  # Every 1 minute
                    rule = StationRule(
                        station_id=idx,
                        station_name=f"Station_{idx+1:03d}",
                        latitude=row['LATITUDE'],
                        longitude=row['LONGITUDE'],
                        expected_time=expected_time,
                        signal_required=True,
                        tolerance_seconds=10.0,
                        tolerance_meters=30.0
                    )
                    rules.append(rule)
            
            print(f"✅ Loaded {len(rules)} station rules")
            return rules
            
        except Exception as e:
            print(f"❌ Error loading station rules: {e}")
            return []
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def find_matching_signal(self, station_rule: StationRule, detected_signals: List[Dict]) -> Optional[Dict]:
        """Find hand signal that matches this station rule"""
        best_match = None
        min_time_diff = float('inf')
        
        for signal in detected_signals:
            if not signal.get('handSignal', {}).get('signal_detected'):
                continue
                
            signal_time = signal.get('timestamp_seconds', 0)
            time_diff = abs(signal_time - station_rule.expected_time)
            
            # Check if within time tolerance
            if time_diff <= station_rule.tolerance_seconds and time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = signal
        
        return best_match
    
    def validate_station_compliance(self, detected_signals: List[Dict], 
                                  current_gps: Optional[Tuple[float, float]] = None) -> List[StationAlert]:
        """Validate compliance for all stations"""
        alerts = []
        
        for rule in self.station_rules:
            # Find matching hand signal
            matching_signal = self.find_matching_signal(rule, detected_signals)
            
            if matching_signal:
                signal_time = matching_signal.get('timestamp_seconds', 0)
                time_diff = signal_time - rule.expected_time
                
                # Determine compliance status
                if abs(time_diff) <= rule.tolerance_seconds:
                    if abs(time_diff) <= 5.0:  # Perfect timing
                        status = 'COMPLIANT'
                        details = f"Perfect timing! Signal raised {abs(time_diff):.1f}s from expected time"
                    else:
                        status = 'COMPLIANT'
                        details = f"Good timing! Signal raised {abs(time_diff):.1f}s {'after' if time_diff > 0 else 'before'} expected time"
                elif time_diff > rule.tolerance_seconds:
                    status = 'LATE'
                    details = f"LATE: Signal raised {time_diff:.1f}s after expected time (tolerance: ±{rule.tolerance_seconds}s)"
                else:
                    status = 'EARLY'
                    details = f"EARLY: Signal raised {abs(time_diff):.1f}s before expected time (tolerance: ±{rule.tolerance_seconds}s)"
                
                # Calculate distance if GPS available
                distance_diff = None
                if current_gps:
                    distance_diff = geodesic(current_gps, (rule.latitude, rule.longitude)).meters
                    if distance_diff > rule.tolerance_meters:
                        status = 'VIOLATION'
                        details += f" | LOCATION ERROR: {distance_diff:.1f}m from station (tolerance: {rule.tolerance_meters}m)"
                
                alert = StationAlert(
                    station_rule=rule,
                    expected_time=self.format_time(rule.expected_time),
                    actual_signal_time=self.format_time(signal_time),
                    status=status,
                    time_difference=time_diff,
                    distance_difference=distance_diff,
                    compliance_details=details
                )
            else:
                # No signal found - MISSED
                alert = StationAlert(
                    station_rule=rule,
                    expected_time=self.format_time(rule.expected_time),
                    actual_signal_time=None,
                    status='MISSED',
                    time_difference=None,
                    distance_difference=None,
                    compliance_details=f"MISSED: No hand signal detected at {rule.station_name} (expected at {self.format_time(rule.expected_time)})"
                )
            
            alerts.append(alert)
        
        return alerts
    
    def generate_station_report(self, alerts: List[StationAlert]) -> Dict:
        """Generate comprehensive station compliance report"""
        total_stations = len(alerts)
        compliant = len([a for a in alerts if a.status == 'COMPLIANT'])
        missed = len([a for a in alerts if a.status == 'MISSED'])
        late = len([a for a in alerts if a.status == 'LATE'])
        early = len([a for a in alerts if a.status == 'EARLY'])
        violations = len([a for a in alerts if a.status == 'VIOLATION'])
        
        compliance_rate = (compliant / total_stations * 100) if total_stations > 0 else 0
        
        return {
            "summary": {
                "total_stations": total_stations,
                "compliant": compliant,
                "missed": missed,
                "late": late,
                "early": early,
                "violations": violations,
                "compliance_rate": compliance_rate
            },
            "alerts": [
                {
                    "station_id": alert.station_rule.station_id,
                    "station_name": alert.station_rule.station_name,
                    "expected_time": alert.expected_time,
                    "actual_time": alert.actual_signal_time,
                    "status": alert.status,
                    "time_difference": alert.time_difference,
                    "distance_difference": alert.distance_difference,
                    "details": alert.compliance_details,
                    "latitude": alert.station_rule.latitude,
                    "longitude": alert.station_rule.longitude
                }
                for alert in alerts
            ]
        }

# Test the system
if __name__ == "__main__":
    system = StationAlertSystem()
    
    # Mock detected signals for testing
    mock_signals = [
        {"timestamp_seconds": 125, "handSignal": {"signal_detected": True}},  # Station 1 - 5s late
        {"timestamp_seconds": 235, "handSignal": {"signal_detected": True}},  # Station 2 - 5s early  
        # Station 3 missing - should be at 360s
        {"timestamp_seconds": 485, "handSignal": {"signal_detected": True}},  # Station 4 - 5s late
    ]
    
    alerts = system.validate_station_compliance(mock_signals)
    report = system.generate_station_report(alerts)
    
    print("\n🚂 STATION ALERT COMPLIANCE REPORT")
    print("=" * 50)
    print(f"Total Stations: {report['summary']['total_stations']}")
    print(f"Compliant: {report['summary']['compliant']}")
    print(f"Missed: {report['summary']['missed']}")
    print(f"Late: {report['summary']['late']}")
    print(f"Early: {report['summary']['early']}")
    print(f"Compliance Rate: {report['summary']['compliance_rate']:.1f}%")
    
    print("\nDetailed Alerts:")
    for alert in alerts[:5]:  # Show first 5
        status_icon = "✅" if alert.status == "COMPLIANT" else "❌"
        print(f"{status_icon} {alert.station_rule.station_name}: {alert.status}")
        print(f"   Expected: {alert.expected_time} | Actual: {alert.actual_signal_time or 'NONE'}")
        print(f"   {alert.compliance_details}")
        print()