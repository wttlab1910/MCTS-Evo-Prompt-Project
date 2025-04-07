"""
Serialization utility for converting objects to and from JSON.
"""
import json
import dataclasses
from typing import Dict, Any, List, Optional, Union, Type, TypeVar, cast
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger("utils.serialization")

T = TypeVar('T')

class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling non-standard types.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable types.
        
        Args:
            obj: Object to convert.
            
        Returns:
            JSON-serializable representation.
        """
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        if isinstance(obj, Enum):
            return obj.value
        
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle custom objects with to_dict method
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return obj.to_dict()
        
        # Let the base class handle other types or raise an error
        return super().default(obj)


def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """
    Convert an object to a JSON string.
    
    Args:
        obj: Object to convert.
        indent: Indentation level for pretty printing.
        
    Returns:
        JSON string representation.
    """
    return json.dumps(obj, cls=JSONEncoder, ensure_ascii=False, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Convert a JSON string to an object.
    
    Args:
        json_str: JSON string to convert.
        
    Returns:
        Python object.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise


def save_json(obj: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save an object as a JSON file.
    
    Args:
        obj: Object to save.
        file_path: Output file path.
        indent: Indentation level for pretty printing.
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, cls=JSONEncoder, ensure_ascii=False, indent=indent)
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load a JSON file.
    
    Args:
        file_path: Input file path.
        
    Returns:
        Python object.
    """
    file_path = Path(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise


def to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """
    Convert a dictionary to a dataclass instance.
    
    Args:
        data: Dictionary to convert.
        cls: Target dataclass type.
        
    Returns:
        Dataclass instance.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    
    # Get field types
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    
    # Convert field values if needed
    field_values = {}
    
    for field_name, field_value in data.items():
        if field_name not in fields:
            continue
        
        field_type = fields[field_name]
        
        # Handle nested dataclasses
        if dataclasses.is_dataclass(field_type) and isinstance(field_value, dict):
            field_values[field_name] = to_dataclass(field_value, field_type)
        else:
            field_values[field_name] = field_value
    
    return cls(**field_values)