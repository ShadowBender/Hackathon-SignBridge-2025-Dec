import sys
import os

print("--- DEBUGGING MEDIAPIPE ---")
print(f"Current Working Directory: {os.getcwd()}")

try:
    import mediapipe as mp
    print(f"1. MediaPipe imported successfully.")
    print(f"2. File location: {mp.__file__}")
    
    # Check for the solutions attribute
    if hasattr(mp, 'solutions'):
        print("3. 'solutions' attribute FOUND. System is working.")
    else:
        print("3. 'solutions' attribute NOT FOUND.")
        print("   CRITICAL: Python is loading the wrong file!")
        print("   It is loading the file listed in step 2 above.")
        print("   DELETE THAT FILE.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print("---------------------------")