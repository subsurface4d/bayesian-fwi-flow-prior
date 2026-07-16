import numpy as np
from typing import List, Optional


class Source(object):
    """Seismic source class.

    Parameters
    ----------
    nt : int
        Number of time samples in the source wavelet
    dt : float
        Time interval of wavelet, in seconds (s)
    f0 : float
        Dominant frequency (Hz) of wavelet for all sources

    Notes: 
        1. The wavelet should be added using the add_source method. The
        provided f0 is for checking numerical dispersion only before launching
        the propagator.
        2. 2-D simulation for now, i.e., (x, z) coordinates.
    """

    def __init__(self, nt: int, dt: float, f0: float) -> None:
        self.nt = nt
        self.dt = dt
        self.f0 = f0

        self.locs = []
        self.type = []
        self.wavelet = []
        self.moment_tensor = []
        self.num = 0

    def __add__(self, other):
        """Add two sources

        Parameters
        ----------
        other : Source
            Another source object

        Returns
        -------
        Source
            A new source object that contains the information of both sources
        """

        if not isinstance(other, Source):
            raise TypeError(
                "Source Error: the other source must be an instance of Source"
            )

        if self.nt != other.nt or self.dt != other.dt:
            raise ValueError(
                "Source Error: the number of time samples and time interval must be the same"
            )

        new_source = Source(self.nt, self.dt, self.f0)
        new_source.locs = self.locs + other.locs
        new_source.type = self.type + other.type
        new_source.wavelet = self.wavelet + other.wavelet
        new_source.moment_tensor = self.moment_tensor + other.moment_tensor
        new_source.num = self.num + other.num

        return new_source

    def __repr__(self):
        """Reimplement the repr function for printing the source information"""

        try:
            locs = np.array(self.locs)
            xmin = locs[:, 0].min()
            xmax = locs[:, 0].max()
            zmin = locs[:, 1].min()
            zmax = locs[:, 1].max()

            info = f"Seismic Source:\n"
            info += f"  Source wavelet: {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            info += f"  Source number : {self.num}\n"
            info += f"  Source types  : {self.get_type(unique = True)}\n"
            info += f"  Source x range: {xmin:6.2f} - {xmax:6.2f} m\n"
            info += f"  Source z range: {zmin:6.2f} - {zmax:6.2f} m\n"
        except:
            info = f"Seismic Source:\n"
            info += f"  empty\n"

        return info

    def add_source(
        self,
        loc: List[float],
        wavelet: np.ndarray,
        type: str,
        mt: Optional[np.ndarray] = None, ) -> None:
        """Append source

        Parameters
        ----------
        loc : List[float]
            Source location in the format of [x, z] (m)
        wvlt : np.ndarray
            Source wavelet, must have the same length as the number of time samples
        type : str
            Source type, can be either pr, vx, vz, mt
        mt : np.ndarray, optional
            Moment tensor (3x3), by default None
        """

        if len(loc) != 2:
            raise ValueError(
                "Source location must be a list in the format of [x, z] (m)"
            )

        if type.lower() not in ["pr", "vx", "vz", "mt"]:
            raise ValueError(
                "Source type must be either pr, vx, vz, or mt"
            )

        if wavelet.shape[0] != self.nt:
            raise ValueError(
                "Source wavelet must have the same length as the number of time samples"
            )

        if mt is not None and mt.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 matrix")

        if type.lower() == "mt" and mt is None:
            raise ValueError("Moment tensor must be provided for mt source")

        # add source
        self.locs.append(loc)
        self.type.append(type)
        self.wavelet.append(wavelet)
        self.moment_tensor.append(mt)
        self.num += 1

    def get_wavelet(self, isrc: int) -> np.ndarray:
        """Return the source wavelet of a given source index as a numpy array

        Parameters
        ----------
        isrc : int
            Source index in the survey from 0 to n_shots - 1

        Returns
        -------
        np.ndarray
            Source wavelet of a given source index as a numpy array
        """

        return self.wavelet[isrc].copy()

    def get_type(self, isrc=None, unique=False) -> List[str]:
        """Return the source type of a given source index as a string

        Parameters
        ----------
        isrc : int
            Source index in the survey from 0 to n_shots - 1

        unqiue : bool
            If True, return the unique source types, i.e., remove the duplicates

        Returns
        -------
        str
            Source type of a given source index as a string
        """

        if isrc is None:
            type = self.type
        else:
            type = self.type[isrc]

        if unique:
            type = list(set(self.type))

        return type

    def get_moment_tensor(self, isrc: int) -> np.ndarray:
        """Return the moment tensor of a given source index as a numpy array

        Parameters
        ----------
        isrc : int
            Source index in the survey from 0 to n_shots - 1

        Returns
        -------
        np.ndarray
            Moment tensor of a given source index as a numpy array
        """

        mt = self.moment_tensor[isrc]

        if mt is None:
            raise RuntimeError(f"Moment tensor is not defined for source {isrc}")

        return mt

    def get_loc(self, isrc=None, type=None) -> np.ndarray:
        """Return all the source coordinates as a numpy array, or a given type

        Parameters
        ----------
        isrc : int, optional
            Source index in the survey from 0 to n_shots - 1, by default None
        type : str, optional
            Source type, can be either pr, vx, vz, or mt, by default None

        Returns
        -------
        np.ndarray
            Source locations of the inquired type or certain source index as a numpy array
        """

        if isrc is None and type is None:
            locs = np.array(self.locs)

        elif isrc is not None and type is None:
            locs = np.array(self.locs[isrc])

        elif isrc is None and type is not None:
            locs = np.array(
                [
                    loc
                    for loc, t in zip(self.locs, self.type)
                    if t.lower() == type.lower()
                ]
            )

        else:
            raise ValueError("Cannot specify both isrc and type")

        if locs.shape[0] == 0:
            locs = np.empty((0, 2), dtype=int)

        return locs
