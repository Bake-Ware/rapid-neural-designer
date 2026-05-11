"""Component and atomic catalog for RND Platform.

Loads component/atomic definitions from web_interface/ JSON files.
Caches in memory after first load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ComponentCatalog:
    def __init__(self, web_interface_root: Path):
        self.root = Path(web_interface_root)
        self._components: dict[str, dict] | None = None
        self._component_index: list[dict] | None = None
        self._atomics: dict[str, list[dict]] | None = None
        self._atomic_categories: dict[str, dict] | None = None

    # ---- Components ----

    def _load_components(self):
        if self._components is not None:
            return
        index_path = self.root / "components" / "index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self._component_index = index["components"]
        self._components = {}
        for entry in self._component_index:
            file_path = self.root / "components" / entry["file"]
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self._components[entry["id"]] = json.load(f)

    def list_components(self) -> list[dict]:
        """Return summary of all components (id, name, description, inputs, outputs, properties)."""
        self._load_components()
        result = []
        for entry in self._component_index:
            comp = self._components.get(entry["id"])
            if not comp:
                continue
            result.append({
                "id": entry["id"],
                "name": comp.get("name", entry["name"]),
                "description": comp.get("description", entry.get("description", "")),
                "color": entry.get("color", ""),
                "inputs": comp.get("inputs", []),
                "outputs": comp.get("outputs", []),
                "properties": comp.get("properties", []),
            })
        return result

    def get_component(self, component_id: str) -> dict | None:
        """Return the full component definition including graph decomposition."""
        self._load_components()
        return self._components.get(component_id)

    # ---- Atomics ----

    def _load_atomics(self):
        if self._atomics is not None:
            return
        index_path = self.root / "atomics" / "index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self._atomic_categories = index["categories"]
        self._atomics = {}
        for cat_id, cat_info in self._atomic_categories.items():
            self._atomics[cat_id] = []
            for filename in cat_info.get("files", []):
                file_path = self.root / "atomics" / filename
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        atoms = json.load(f)
                        if isinstance(atoms, list):
                            self._atomics[cat_id].extend(atoms)

    def list_atomic_categories(self) -> list[dict]:
        """Return list of atomic categories."""
        self._load_atomics()
        return [
            {"id": cat_id, "label": info["label"], "icon": info.get("icon", ""),
             "count": len(self._atomics.get(cat_id, []))}
            for cat_id, info in self._atomic_categories.items()
        ]

    def list_atomics(self, category: str = "") -> list[dict]:
        """Return atomics, optionally filtered by category. Summary format."""
        self._load_atomics()
        result = []
        cats = [category] if category else list(self._atomics.keys())
        for cat_id in cats:
            for atom in self._atomics.get(cat_id, []):
                result.append({
                    "id": atom["id"],
                    "name": atom["name"],
                    "category": atom.get("category", cat_id),
                    "description": atom.get("description", ""),
                    "inputs": atom.get("inputs", []),
                    "outputs": atom.get("outputs", []),
                    "properties": atom.get("properties", []),
                })
        return result

    def get_atomic(self, category: str, atomic_id: str) -> dict | None:
        """Return the full atomic definition."""
        self._load_atomics()
        for atom in self._atomics.get(category, []):
            if atom["id"] == atomic_id:
                return atom
        return None

    # ---- Promote (write to codebase) ----

    def promote_component(self, comp_id: str, definition: dict) -> Path:
        """Write a component definition to the codebase and update index.json."""
        self._load_components()
        # Write the component file
        file_name = f"{comp_id}.json"
        file_path = self.root / "components" / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(definition, f, indent=2, ensure_ascii=False)

        # Update index.json
        index_path = self.root / "components" / "index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        # Check if already in index
        existing_ids = {c["id"] for c in index["components"]}
        if comp_id not in existing_ids:
            index["components"].append({
                "id": comp_id,
                "name": definition.get("name", comp_id),
                "description": definition.get("description", ""),
                "file": file_name,
                "color": definition.get("color", "#888888"),
            })
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        # Invalidate cache
        self._components = None
        self._component_index = None
        return file_path

    def promote_atomic(self, definition: dict, category: str) -> Path:
        """Append an atomic definition to the appropriate category file and update index."""
        self._load_atomics()
        if category not in self._atomic_categories:
            raise ValueError(f"Unknown atomic category: {category}")

        # Find the file for this category
        cat_info = self._atomic_categories[category]
        file_name = cat_info["files"][0]  # Use first file
        file_path = self.root / "atomics" / file_name

        # Load existing atomics
        with open(file_path, "r", encoding="utf-8") as f:
            atoms = json.load(f)

        # Check if already exists
        existing_ids = {a["id"] for a in atoms}
        if definition["id"] not in existing_ids:
            atoms.append(definition)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(atoms, f, indent=2, ensure_ascii=False)

        # Invalidate cache
        self._atomics = None
        self._atomic_categories = None
        return file_path

    def get_node_definition(self, node_type: str) -> dict | None:
        """Look up a node definition by its type string.
        Format: 'molecular/{component_id}' or 'atomic/{category}/{atomic_id}'
        Returns the definition dict with inputs, outputs, properties."""
        parts = node_type.split("/")
        if parts[0] == "molecular" and len(parts) == 2:
            return self.get_component(parts[1])
        elif parts[0] == "atomic" and len(parts) == 3:
            return self.get_atomic(parts[1], parts[2])
        return None
