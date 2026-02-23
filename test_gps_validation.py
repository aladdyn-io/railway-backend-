#!/usr/bin/env python3
"""
Test script for GPS-enabled hand signal detection
Demonstrates violation detection when signals are raised at incorrect locations
"""

import requests
import json

# API base URL
API_BASE = "http://localhost:9001"

def test_gps_validation():
    """Test GPS validation functionality"""
    
    print("🧪 Testing GPS Validation for Hand Signals")
    print("=" * 50)
    
    # 1. Get valid locations from Excel
    print("\n1. Fetching valid signal locations...")
    try:
        response = requests.get(f"{API_BASE}/api/valid-locations")
        if response.status_code == 200:
            data = response.json()
            locations = data['locations']
            print(f"✅ Found {len(locations)} valid locations")
            
            # Show first few locations
            for i, loc in enumerate(locations[:3]):
                print(f"   Location {i+1}: {loc['latitude']:.6f}, {loc['longitude']:.6f}")
            
            if len(locations) > 3:
                print(f"   ... and {len(locations)-3} more locations")
        else:
            print(f"❌ Failed to get locations: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error fetching locations: {e}")
        return
    
    # 2. Test valid location (should pass)
    print("\n2. Testing VALID location (should pass)...")
    valid_location = locations[0]  # Use first valid location
    test_data = {
        "latitude": valid_location['latitude'],
        "longitude": valid_location['longitude']
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/set-gps", json=test_data)
        if response.status_code == 200:
            print(f"✅ Set GPS to valid location: {test_data['latitude']:.6f}, {test_data['longitude']:.6f}")
            print("   → Hand signals at this location should be COMPLIANT")
        else:
            print(f"❌ Failed to set GPS: {response.status_code}")
    except Exception as e:
        print(f"❌ Error setting GPS: {e}")
    
    # 3. Test invalid location (should fail)
    print("\n3. Testing INVALID location (should trigger violation)...")
    invalid_location = {
        "latitude": valid_location['latitude'] + 0.01,  # Move ~1km away
        "longitude": valid_location['longitude'] + 0.01
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/set-gps", json=invalid_location)
        if response.status_code == 200:
            print(f"✅ Set GPS to invalid location: {invalid_location['latitude']:.6f}, {invalid_location['longitude']:.6f}")
            print("   → Hand signals at this location should be VIOLATIONS")
        else:
            print(f"❌ Failed to set GPS: {response.status_code}")
    except Exception as e:
        print(f"❌ Error setting GPS: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 GPS Validation Test Complete!")
    print("\nNow when you upload a video:")
    print("• Hand signals will be validated against Excel locations")
    print("• Violations will be flagged when signals are raised at wrong locations")
    print("• Events will show GPS validation status and distance info")
    print("• Alert count will include hand signal violations")

def show_excel_data():
    """Display Excel data structure"""
    print("\n📊 Excel File Data Structure:")
    print("=" * 30)
    
    try:
        import pandas as pd
        df = pd.read_excel("Detected_Signals_Lat_Long.xlsx")
        print(f"Total valid locations: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nSample locations:")
        print(df.head().to_string(index=False))
        
        # Calculate coverage area
        lat_range = df['LATITUDE'].max() - df['LATITUDE'].min()
        lon_range = df['LONGITUDE'].max() - df['LONGITUDE'].min()
        print(f"\nCoverage area:")
        print(f"Latitude range: {lat_range:.6f}° ({lat_range*111:.1f}km)")
        print(f"Longitude range: {lon_range:.6f}° ({lon_range*111:.1f}km)")
        
    except Exception as e:
        print(f"Error reading Excel: {e}")

if __name__ == "__main__":
    print("🚂 Railway Hand Signal GPS Validation Test")
    print("=" * 50)
    
    # Show Excel data first
    show_excel_data()
    
    # Test API endpoints
    test_gps_validation()
    
    print("\n🔧 Usage Instructions:")
    print("1. Start the API: python3 video_processor_api.py")
    print("2. Upload a video through the frontend")
    print("3. Hand signals will be validated against GPS locations")
    print("4. Check events for violation status and GPS info")