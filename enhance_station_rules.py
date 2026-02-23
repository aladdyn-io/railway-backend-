"""
Enhance Detected_Signals_Lat_Long.xlsx with timing information
This script adds EXPECTED_TIME_SECONDS and other required columns
"""
import pandas as pd
import numpy as np

def enhance_station_rules(input_file="Detected_Signals_Lat_Long.xlsx", output_file="Detected_Signals_Lat_Long_Enhanced.xlsx"):
    """Add timing and station information to existing Excel file"""
    
    try:
        # Read existing file
        df = pd.read_excel(input_file)
        print(f"✅ Loaded {len(df)} stations from {input_file}")
        print(f"📋 Existing columns: {df.columns.tolist()}")
        
        # Check if already enhanced
        if 'EXPECTED_TIME_SECONDS' in df.columns:
            print("✅ File already has timing information!")
            return output_file
        
        # Add required columns
        enhanced_data = []
        
        for idx, row in df.iterrows():
            # Calculate expected time - you can customize this logic
            # Option 1: Every 60 seconds per station
            expected_time_seconds = idx * 60
            
            # Option 2: Based on distance (if you have distance data)
            # expected_time_seconds = calculate_time_from_distance(...)
            
            # Format time
            hours = int(expected_time_seconds // 3600)
            minutes = int((expected_time_seconds % 3600) // 60)
            seconds = int(expected_time_seconds % 60)
            expected_time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            enhanced_row = {
                'STATION_ID': idx + 1,
                'STATION_NAME': f"Station_{idx+1:03d}",  # You can customize names
                'LATITUDE': row['LATITUDE'],
                'LONGITUDE': row['LONGITUDE'],
                'EXPECTED_TIME_SECONDS': expected_time_seconds,
                'EXPECTED_TIME_FORMATTED': expected_time_formatted,
                'SIGNAL_REQUIRED': True,  # All stations require signals
                'TOLERANCE_SECONDS': 10.0,  # ±10 seconds tolerance
                'TOLERANCE_METERS': 30.0,  # ±30 meters GPS tolerance
                'PRIORITY': 'HIGH'  # Default priority
            }
            
            enhanced_data.append(enhanced_row)
        
        # Create enhanced DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Save to new file
        enhanced_df.to_excel(output_file, index=False)
        
        print(f"\n✅ ENHANCED STATION RULES CREATED!")
        print("=" * 60)
        print(f"📁 Output file: {output_file}")
        print(f"📊 Total stations: {len(enhanced_df)}")
        print(f"\n📋 New columns added:")
        print("   • STATION_ID")
        print("   • STATION_NAME")
        print("   • EXPECTED_TIME_SECONDS")
        print("   • EXPECTED_TIME_FORMATTED")
        print("   • SIGNAL_REQUIRED")
        print("   • TOLERANCE_SECONDS")
        print("   • TOLERANCE_METERS")
        print("   • PRIORITY")
        
        print(f"\n🎯 Sample data:")
        print(enhanced_df.head(5)[['STATION_ID', 'STATION_NAME', 'EXPECTED_TIME_FORMATTED', 'LATITUDE', 'LONGITUDE']].to_string(index=False))
        
        print(f"\n⏰ Timing Information:")
        print(f"   • First station: {enhanced_df.iloc[0]['EXPECTED_TIME_FORMATTED']} (0 seconds)")
        print(f"   • Second station: {enhanced_df.iloc[1]['EXPECTED_TIME_FORMATTED']} (60 seconds)")
        print(f"   • Last station: {enhanced_df.iloc[-1]['EXPECTED_TIME_FORMATTED']} ({enhanced_df.iloc[-1]['EXPECTED_TIME_SECONDS']} seconds)")
        print(f"   • Total duration: {enhanced_df.iloc[-1]['EXPECTED_TIME_SECONDS'] / 60:.1f} minutes")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the timing in {output_file}")
        print(f"   2. Update EXPECTED_TIME_SECONDS with actual video timestamps")
        print(f"   3. Update STATION_NAME with real station names")
        print(f"   4. Update the code to use: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚂 ENHANCING STATION RULES WITH TIMING INFORMATION")
    print("=" * 60)
    
    output = enhance_station_rules()
    
    if output:
        print(f"\n✅ Success! Enhanced file saved as: {output}")
        print("📝 Update video_processor_api.py to use this file")
    else:
        print("\n❌ Failed to create enhanced file")
