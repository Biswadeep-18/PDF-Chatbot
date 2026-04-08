import json
import os
import base64
from typing import Dict, Any

class ProfileService:
    def __init__(self, storage_dir: str = "user_data"):
        self.storage_path = os.path.join(storage_dir, "profile.json")
        self.image_dir = os.path.join(storage_dir, "images")
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            
        self.default_profile = {
            "name": "User",
            "email": "user@example.com",
            "profile_image": None,
            "theme": "light"
        }

    def get_profile(self) -> Dict[str, Any]:
        if not os.path.exists(self.storage_path):
            return self.default_profile
        
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except:
            return self.default_profile

    def update_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        profile = self.get_profile()
        profile.update(data)
        
        with open(self.storage_path, "w") as f:
            json.dump(profile, f, indent=2)
        
        return profile

    def save_image(self, image_data_b64: str, filename: str) -> str:
        """Save base64 image data and return the local path"""
        if not image_data_b64:
            return None
            
        # Remove header if present
        if "," in image_data_b64:
            image_data_b64 = image_data_b64.split(",")[1]
            
        file_path = os.path.join(self.image_dir, filename)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data_b64))
            
        return file_path
