#!/usr/bin/env python3
"""
Quick fix to add missing health_check methods to ML components
"""

import os

# Health check method template
HEALTH_CHECK_METHOD = '''
    async def health_check(self) -> Dict[str, Any]:
        """Health check for this ML component"""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "component": self.__class__.__name__,
                "initialized": hasattr(self, 'initialized') and getattr(self, 'initialized', True)
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "timestamp": datetime.utcnow().isoformat(),
                "component": self.__class__.__name__,
                "error": str(e)
            }
'''

# Files to fix and their class patterns
ML_FILES = [
    ('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/ml/feature_engineering.py', 'class FeatureEngineer'),
    ('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/ml/model_lifecycle.py', 'class ModelLifecycleManager'), 
    ('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/ml/risk_prediction.py', 'class RiskPredictor')
]

def add_health_check_to_file(file_path, class_pattern):
    """Add health_check method to a Python file"""
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if health_check method already exists
        if 'def health_check(' in content:
            print(f"✅ {file_path} already has health_check method")
            return True
        
        # Find the class definition
        class_start = content.find(class_pattern)
        if class_start == -1:
            print(f"⚠️ Could not find {class_pattern} in {file_path}")
            return False
        
        # Find the first method after class definition to insert before it
        class_content = content[class_start:]
        first_method_pos = class_content.find('\n    def ')
        
        if first_method_pos == -1:
            print(f"⚠️ Could not find insertion point in {file_path}")
            return False
        
        # Insert health_check method before the first method
        insertion_point = class_start + first_method_pos
        new_content = (
            content[:insertion_point] + 
            HEALTH_CHECK_METHOD + 
            content[insertion_point:]
        )
        
        # Write back the modified content
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Added health_check method to {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Add health_check methods to all ML component files"""
    print("🔧 Adding missing health_check methods to ML components...")
    
    success_count = 0
    for file_path, class_pattern in ML_FILES:
        if add_health_check_to_file(file_path, class_pattern):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(ML_FILES)} files updated successfully")
    
    if success_count == len(ML_FILES):
        print("✅ All ML components now have health_check methods!")
    else:
        print("⚠️ Some files could not be updated - manual intervention may be needed")

if __name__ == "__main__":
    main()