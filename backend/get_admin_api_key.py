#!/usr/bin/env python3
"""
Simple script to get the admin user's API key for testing
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth.database import user_db

def get_admin_api_key():
    """Get the admin user's API key"""
    try:
        # Get admin user
        admin_user = user_db.get_user_by_username("admin")
        if not admin_user:
            print("ERROR: Admin user not found!")
            return None
        
        # Get the API key from the internal user dict
        admin_dict = user_db._users[admin_user.id]
        api_key = admin_dict["api_key"]
        
        print(f"Admin API Key: {api_key}")
        print(f"Username: {admin_user.username}")
        print(f"User ID: {admin_user.id}")
        
        return api_key
        
    except Exception as e:
        print(f"ERROR: Failed to get API key: {e}")
        return None

if __name__ == "__main__":
    get_admin_api_key()