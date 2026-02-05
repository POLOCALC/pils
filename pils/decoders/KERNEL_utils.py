import pickle
import struct
from typing import Any, Dict

from ..utils.logging_config import get_logger

from . import KERNEL_dicts as Kdb

logger = get_logger(__name__)

HEADER = b"\xaa\x55"


def _checksum(msg: bytes) -> bytes:
    """Compute the checksum of a message.

    The checksum is the arithmetical sum of all the bytes in the message
    excluding the header. The resulting sum is an unsigned short integer
    whose Least Significant Byte is first.

    Parameters
    ----------
    msg : bytes
        Message bytes used to compute the checksum.

    Returns
    -------
    bytes
        Checksum as bytes (2 bytes, little-endian).
    """

    if msg.startswith(HEADER):
        msg = msg[2:]

    return sum(msg).to_bytes(2, byteorder="little", signed=False)


class KernelMsg:
    """Decoder for KERNEL inclinometer messages."""

    def __init__(self) -> None:

        self.msg_address = []

        for i in Kdb.MODES.keys():
            self.msg_address.append(Kdb.MODES[i]["Address"])

    def decode_single(self, msg: bytes, return_dict: bool = False) -> Dict[str, Any]:
        """Decode a single message sent by the inclinometer.

        The structure of the message is presented in the KERNEL IMU ICD v1.27.

        Parameters
        ----------
        msg : bytes
            Message bytes to be decoded.
        return_dict : bool, optional
            If True, return as dictionary (currently unused).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing decoded message fields.
        """

        if msg[:2] == HEADER:
            type_idx = 3
        else:
            type_idx = 1

        msg_type = msg[type_idx].to_bytes(1, byteorder="little")
        modes = list(Kdb.MODES.keys())

        vals = {}

        idx = self.msg_address.index(msg_type)

        vals["Type"] = modes[idx]

        start = type_idx + 3

        try:
            for i in range(len(Kdb.MODES[modes[idx]]["Type"])):
                # mm = Kdb.MODES[modes[idx]]["Parameters"][i]  # noqa: F841 (unused)

                fmt = "<" + "".join(Kdb.MODES[modes[idx]]["Type"][i])
                val = msg[start : start + struct.calcsize(Kdb.MODES[modes[idx]]["Type"][i])]

                if Kdb.MODES[modes[idx]]["Parameters"][i] != "USW":
                    (tmp,) = struct.unpack(fmt, val)
                    vals[Kdb.MODES[modes[idx]]["Parameters"][i]] = (
                        tmp / Kdb.MODES[modes[idx]]["Scale"][i]
                    )
                else:
                    tmp = Kdb.extract_USW(val)
                    vals[Kdb.MODES[modes[idx]]["Parameters"][i]] = tmp

                start += struct.calcsize(Kdb.MODES[modes[idx]]["Type"][i])
        except KeyError:
            pass

        return vals

    def decode_multi(self, filename: str) -> Dict[str, list]:
        """Decode multiple messages saved in a binary file.

        Parameters
        ----------
        filename : str
            Path to binary file containing KERNEL messages.

        Returns
        -------
        Dict[str, list]
            Dictionary with parameter names as keys and lists of decoded values.
        """
        count = 0

        with open(filename, "rb") as fd:
            if filename[-3:].lower() == ".pck":
                data = pickle.load(fd)
            else:
                data = fd.read()

        logger.info(f"Decoded {len(data)} values")

        parts = data.split(HEADER)

        # Reattach the header to each split part (except the first, which was before the first header)
        messages = [HEADER + part for part in parts[1:]]

        data = data[data.find(HEADER) :]

        decoded = {}
        count = 0

        for msg in messages:
            try:
                tmp = self.decode_single(msg, return_dict=True)

                for j in tmp.keys():
                    if count == 0:
                        decoded[j] = []

                    decoded[j].append(tmp[j])

            except struct.error:
                pass
            except ValueError:
                pass
            except IndexError:
                pass

            count += 1

        return decoded
