"""Kernel-100 inclinometer message decoder."""

import struct
from typing import Any

from ..utils.logging_config import get_logger
from . import KERNEL_dicts as Kdb

logger = get_logger(__name__)

HEADER = b"\xaa\x55"


class KernelMsg:
    """Decoder for KERNEL inclinometer messages."""

    def __init__(self) -> None:
        self.msg_address = [mode["Address"] for mode in Kdb.MODES.values()]

    def decode_single(self, msg: bytes) -> dict[str, Any]:
        """Decode a single message sent by the inclinometer.

        The structure of the message is presented in the KERNEL IMU ICD v1.27.

        Parameters
        ----------
        msg : bytes
            Message bytes to be decoded, with or without the header.

        Returns
        -------
        Dict[str, Any]
            The message's ``Type`` and its decoded fields.
        """
        type_idx = 3 if msg[:2] == HEADER else 1
        msg_type = msg[type_idx].to_bytes(1, byteorder="little")
        name = list(Kdb.MODES)[self.msg_address.index(msg_type)]
        mode = Kdb.MODES[name]
        vals: dict[str, Any] = {"Type": name}

        start = type_idx + 3
        try:
            for i, code in enumerate(mode["Type"]):
                size = struct.calcsize(code)
                val = msg[start : start + size]
                param = mode["Parameters"][i]
                if param == "USW":
                    vals[param] = Kdb.extract_USW(val)
                else:
                    (raw,) = struct.unpack("<" + code, val)
                    vals[param] = raw / mode["Scale"][i]
                start += size
        except KeyError:  # a mode without a field layout: only its Type
            pass

        return vals
