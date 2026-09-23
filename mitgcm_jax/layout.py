"""Array layout of every model field and Fortran-index helpers (plan: Technical Details).

Fields are stored per tile with halos, exactly like MITgcm's `(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, [Nr,] nSx*nSy)` arrays,
but with the tile axis first and x fastest: shape `[tile, j, i]` (2-D) or `[tile, k, j, i]` (3-D), float64.
Tile t (0-based) is exch2 tile number t+1 (W2 numbering of `data.exch2`; for 13x90x90: 1-3 facet 1, 4-6 facet 2,
7 facet 3, 8-10 facet 4, 11-13 facet 5, the order used by the PO.DAAC netCDF products).

A Fortran index i (1-based, halo points <= 0 or > sNx) sits at Python index i - 1 + OLx; k (1..Nr) at k - 1.
Kernels are literal translations of Fortran loops: `DO j=jMin,jMax; DO i=iMin,iMax` becomes a slice
`L.js(jMin, jMax), L.is_(iMin, iMax)` and a shifted access `a(i-1,j)` becomes `L.is_(iMin-1, iMax-1)`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    sNx: int = 90
    sNy: int = 90
    OLx: int = 4
    OLy: int = 4
    nTiles: int = 13
    Nr: int = 50

    @property
    def nx(self):  # padded extent in x (1-OLx .. sNx+OLx)
        return self.sNx + 2 * self.OLx

    @property
    def ny(self):
        return self.sNy + 2 * self.OLy

    def is_(self, lo, hi):
        """Python slice of Fortran i-range lo..hi (inclusive)."""
        return slice(lo - 1 + self.OLx, hi + self.OLx)

    def js(self, lo, hi):
        """Python slice of Fortran j-range lo..hi (inclusive)."""
        return slice(lo - 1 + self.OLy, hi + self.OLy)

    def ii(self, i):
        """Python index of Fortran i."""
        return i - 1 + self.OLx

    def jj(self, j):
        return j - 1 + self.OLy

    @property
    def shape2d(self):
        return (self.nTiles, self.ny, self.nx)

    @property
    def shape3d(self):
        return (self.nTiles, self.Nr, self.ny, self.nx)


LLC90_13 = Layout()
