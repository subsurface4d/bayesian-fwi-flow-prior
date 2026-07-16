from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Receiver(object):
    """Seismic receiver class

    Parameters
    ----------
    nt : int
        Number of time samples in the receiver data
    dt : float
        Time interval (s) of data

    Notes: 1. The seismic data is assumed to start at time 0, e.g., ot = 0.
           2. The receiver locations should be added using the add_receiver method.
           3. The DAS cable should be added using the add_cable method.
           4. The receiver is assumed to be the same for all shots, i.e., the
             receiver locations are not shot-dependent. This consideration is
             based on the fact the regular data (nshot, nrec, nt) can be dealt
             more efficiently on GPUs than irregular data. However, this may 
             limit the flexibility of modeling streamer data where the receiver
             locations are shot-dependent. The workaround solution is to apply 
             offset masking to the regular data to mimic the streamer data.
    """

    def __init__(self, nt: int, dt: float) -> None:
        self.nt = nt
        self.dt = dt

        self.locs = []
        self.type = []
        self.num = 0
        self.cable = None

    def __add__(self, other: "Receiver") -> "Receiver":
        """Add two receivers

        Parameters
        ----------
        other : Receiver
            Another receiver object

        Returns
        -------
        Receiver
            A new receiver object that contains the information of both receivers
        """

        if not isinstance(other, Receiver):
            raise TypeError(
                "Receiver Error: the other receiver must be an instance of Receiver"
            )

        if self.nt != other.nt or self.dt != other.dt:
            raise ValueError(
                "Receiver Error: the number of time samples and time interval must be the same"
            )

        new_receiver = Receiver(self.nt, self.dt)
        new_receiver.locs = self.locs + other.locs
        new_receiver.type = self.type + other.type
        new_receiver.num = self.num + other.num

        # addition of the cable is ignored for now

        return new_receiver

    def __repr__(self):
        """Print the receiver information
        """

        try:
            locs = np.array(self.locs)
            xmin = locs[:, 0].min()
            xmax = locs[:, 0].max()
            zmin = locs[:, 1].min()
            zmax = locs[:, 1].max()

            info = f"Seismic Receiver:\n"
            info += (
                f"  Receiver data   : {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            )
            info += f"  Receiver number : {self.num}\n"
            info += f"  Receiver types  : {self.get_type(unique = True)}\n"
            info += f"  Receiver x range: {xmin:6.2f} - {xmax:6.2f} m\n"
            info += f"  Receiver z range: {zmin:6.2f} - {zmax:6.2f} m\n"
        except:
            info = f"Seismic Receiver:\n"
            info += f"  empty\n"

        return info

    def add_receiver(self, loc: List[float], type: str) -> None:
        """Append receiver object to the survey

        Parameters
        ----------
        loc : List[float]
            Receiver coordinates in the format of [x, z] (m)
        type : str
            Receiver type, can be either pr, vx, or vz
        """

        if len(loc) != 2:
            raise ValueError(
                "Receiver Error: the location must be in the format of [x, z]"
            )

        if type.lower() not in ["pr", "vx", "vz", "das"]:
            raise ValueError("Receiver type must be either pr, vx, vz, or das")

        # add the receiver
        self.locs.append(loc)
        self.type.append(type)
        self.num += 1
        self.cable_num = 0

    def get_loc(self, type: str) -> np.ndarray:
        """Return the receiver (or a given type) locations as a numpy array

        Parameters
        ----------
        type : str
            Receiver type, can be either pr, vx, vz, or das

        Returns
        -------
        np.ndarray
            Receiver of a given type locations as a numpy array
        """

        if type.lower() not in ["pr", "vx", "vz", "das"]:
            raise ValueError("Receiver type must be either pr, vx, vz, or das")

        rec_locs = np.array(
            [loc for loc, t in zip(self.locs, self.type) if t.lower() == type.lower()]
        )

        if rec_locs.shape[0] == 0:
            rec_locs = np.empty((0, 2), dtype=int)

        return rec_locs

    def get_type(self, unique=False) -> List[str]:
        """Return all the receiver types as a list

        Parameters
        ----------
        unique : bool, optional
            Return unique receiver types, by default False

        Returns
        -------
        List[str]
            All receiver types as a list
        """

        if unique:
            return list(set(self.type))
        else:
            return self.type
        
    def get_loc_dict(self) -> dict:
        """Return the receiver locations as a dictionary

        Returns
        -------
        dict
            Receiver locations as a dictionary
        """

        loc_dict = {}
        for loc, t in zip(self.locs, self.type):
            if t.lower() not in loc_dict.keys():
                loc_dict[t.lower()] = []
            loc_dict[t.lower()].append(loc)

        return loc_dict
