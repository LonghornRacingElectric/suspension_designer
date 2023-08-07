"""utilities.py - Assorted Helper Functions"""
import typing as typ

__all__ = ['ordered_unique', 'sequence_to_index']

def ordered_unique(col: typ.Collection) -> tuple:
    """Returns order preserved unique set"""
    seen = set()
    return tuple(ele for ele in col if not (ele in seen or seen.add(ele)))

def sequence_to_index(value: str) -> list[int]: 
    """Converts cardinal axis string sequence to """
    mapping = {'x': 0, 'y': 1, 'z': 2}
    return [mapping[c] for c in value.lower()]

