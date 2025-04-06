"""
Serialization utility for MCTS-Evo-Prompt system.
"""
import json
import pickle
import base64
from app.utils.logger import get_logger

logger = get_logger("serialization")

def dumps(obj):
    """
    Serialize an object to a string.
    
    Args:
        obj: Object to serialize.
        
    Returns:
        Serialized string.
    """
    try:
        # Try to serialize with json first
        return json.dumps(obj)
    except (TypeError, OverflowError):
        # If json fails, use pickle with base64 encoding
        try:
            pickle_data = pickle.dumps(obj)
            return {
                "__pickle__": True,
                "data": base64.b64encode(pickle_data).decode('ascii')
            }
        except Exception as e:
            logger.error(f"Failed to serialize object: {e}")
            raise

def loads(serialized_obj):
    """
    Deserialize an object from a string.
    
    Args:
        serialized_obj: Serialized string.
        
    Returns:
        Deserialized object.
    """
    if isinstance(serialized_obj, dict) and serialized_obj.get("__pickle__"):
        try:
            pickle_data = base64.b64decode(serialized_obj["data"])
            return pickle.loads(pickle_data)
        except Exception as e:
            logger.error(f"Failed to deserialize object: {e}")
            raise
    else:
        try:
            return serialized_obj
        except Exception as e:
            logger.error(f"Failed to deserialize JSON object: {e}")
            raise