import pandas as pd
import numpy as np

# Create the EXACT Excel structure needed for perfect station alert detection
station_data = {
    'STATION_ID': range(1, 21),  # 20 stations for demo
    'STATION_NAME': [
        'Central Station', 'North Junction', 'East Terminal', 'West Bridge', 'South Gate',
        'Industrial Zone', 'City Center', 'Harbor View', 'Mountain Pass', 'Valley Station',
        'River Crossing', 'Forest Junction', 'Desert Stop', 'Coastal Terminal', 'Highland Station',
        'Lakeside Platform', 'Suburban Hub', 'Metro Junction', 'Airport Link', 'Final Destination'
    ],
    'LATITUDE': [
        15.267390, 15.267407, 15.267404, 15.267402, 15.267413,
        15.267420, 15.267435, 15.267450, 15.267465, 15.267480,
        15.267495, 15.267510, 15.267525, 15.267540, 15.267555,
        15.267570, 15.267585, 15.267600, 15.267615, 15.267630
    ],
    'LONGITUDE': [
        73.980355, 73.980444, 73.980442, 73.980444, 73.980433,
        73.980450, 73.980465, 73.980480, 73.980495, 73.980510,
        73.980525, 73.980540, 73.980555, 73.980570, 73.980585,
        73.980600, 73.980615, 73.980630, 73.980645, 73.980660
    ],
    'EXPECTED_TIME_SECONDS': [
        30, 90, 150, 210, 270,      # Every 60 seconds (1 minute intervals)
        330, 390, 450, 510, 570,
        630, 690, 750, 810, 870,
        930, 990, 1050, 1110, 1170
    ],
    'EXPECTED_TIME_FORMATTED': [
        '00:00:30', '00:01:30', '00:02:30', '00:03:30', '00:04:30',
        '00:05:30', '00:06:30', '00:07:30', '00:08:30', '00:09:30',
        '00:10:30', '00:11:30', '00:12:30', '00:13:30', '00:14:30',
        '00:15:30', '00:16:30', '00:17:30', '00:18:30', '00:19:30'
    ],
    'SIGNAL_REQUIRED': [True] * 20,  # All stations require hand signals
    'TOLERANCE_SECONDS': [10] * 20,  # ±10 seconds tolerance for all stations
    'PRIORITY': ['HIGH', 'HIGH', 'MEDIUM', 'HIGH', 'MEDIUM',
                'LOW', 'HIGH', 'MEDIUM', 'LOW', 'HIGH',
                'MEDIUM', 'LOW', 'HIGH', 'MEDIUM', 'HIGH',
                'LOW', 'MEDIUM', 'HIGH', 'HIGH', 'CRITICAL']
}

# Create DataFrame
df = pd.DataFrame(station_data)

# Save to Excel
excel_filename = 'Station_Alert_Rules.xlsx'
df.to_excel(excel_filename, index=False)

print("✅ PERFECT STATION ALERT EXCEL FILE CREATED!")
print("=" * 60)
print(f"📁 File: {excel_filename}")
print(f"📊 Stations: {len(df)}")
print("\n📋 Excel Structure:")
for col in df.columns:
    print(f"   • {col}")

print("\n🎯 Sample Data:")
print(df.head(5).to_string(index=False))

print("\n🚂 HOW IT WORKS:")
print("=" * 30)
print("1. Video starts at 00:00:00")
print("2. At 00:00:30 - Pilot should raise hand at Central Station")
print("3. At 00:01:30 - Pilot should raise hand at North Junction")
print("4. At 00:02:30 - Pilot should raise hand at East Terminal")
print("5. System checks if hand signals match these exact times")
print("6. ±10 seconds tolerance (00:00:20 to 00:00:40 is OK for station 1)")

print("\n✅ COMPLIANCE DETECTION:")
print("• COMPLIANT: Hand raised within ±10 seconds of expected time")
print("• MISSED: No hand signal detected at expected time")
print("• LATE: Hand signal raised >10 seconds after expected time")
print("• EARLY: Hand signal raised >10 seconds before expected time")

print(f"\n🎯 Use this file: {excel_filename}")
print("Replace 'Detected_Signals_Lat_Long.xlsx' with this file for perfect station alerts!")