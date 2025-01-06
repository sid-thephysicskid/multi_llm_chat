import json
import os
from pathlib import Path
from typing import Dict, Optional

class PromptManager:
    def __init__(self):
        self.prompts_dir = Path("saved_prompts")
        self.prompts_file = self.prompts_dir / "prompts.json"
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize the storage directory and file if they don't exist."""
        self.prompts_dir.mkdir(exist_ok=True)
        if not self.prompts_file.exists():
            self._save_prompts({})

    def _save_prompts(self, prompts: Dict):
        """Save prompts to the JSON file."""
        with open(self.prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)

    def _load_prompts(self) -> Dict:
        """Load prompts from the JSON file."""
        if self.prompts_file.exists():
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        return {}

    def save_prompt(self, name: str, system_prompt: str) -> bool:
        """Save a new prompt with the given name."""
        prompts = self._load_prompts()
        prompts[name] = system_prompt
        self._save_prompts(prompts)
        return True

    def get_prompt(self, name: str) -> Optional[str]:
        """Retrieve a prompt by name."""
        prompts = self._load_prompts()
        return prompts.get(name)

    def list_prompts(self) -> Dict[str, str]:
        """List all saved prompts."""
        return self._load_prompts()

    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt by name."""
        prompts = self._load_prompts()
        if name in prompts:
            del prompts[name]
            self._save_prompts(prompts)
            return True
        return False 