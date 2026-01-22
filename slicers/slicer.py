from typing import List


class BaseSlicer:
    """
    Contract:
      - .slice() -> returns the next slice geometry (as PyVista PolyData) or None when done
      - .num_slices -> total slices
      - .zs -> z value per slice index
    """

    def slice(self):
        raise NotImplementedError

    @property
    def num_slices(self) -> int:
        raise NotImplementedError

    @property
    def zs(self) -> List[float]:
        raise NotImplementedError
