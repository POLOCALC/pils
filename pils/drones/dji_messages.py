"""Decoding single DJI .dat messages (GPS records) one at a time."""

import datetime
import struct
from typing import Any

from ..utils.logging_config import get_logger
from .dji_dat import MESSAGE_DEFINITIONS, calc_checksum

logger = get_logger(__name__)


class DJIMessageMixin:
    """Decoding of single .dat messages for :class:`~pils.drones.DJIDrone.DJIDrone`."""

    def _parse_and_decode_message(self, msg_data: bytes) -> list[dict[str, Any]]:
        """Parse and decode a single message.

        Message structure:
        - Byte 0: 0x55 (marker)
        - Byte 1: Message length
        - Bytes 2-4: Reserved
        - Bytes 5-6: Message type (little-endian uint16)
        - Byte 6: XOR key
        - Bytes 8-11: Tick (timestamp in ticks, little-endian uint32)
        - Bytes 12+: Encrypted payload

        Parameters
        ----------
        msg_data : bytes
            Raw message bytes to parse.

        Returns
        -------
        List[Dict[str, Any]]
            List containing decoded message dictionary, or empty list if parsing fails.
        """
        if len(msg_data) < 12:
            return []
        # Handle case where message doesn't start with 0x55
        if msg_data[0] != 0x55:
            return []

        try:
            # Extract header fields
            msg_length = msg_data[1]
            msg_type = struct.unpack("<H", msg_data[4:6])[0]
            key = msg_data[6]
            tick_val = struct.unpack("<I", msg_data[6:10])[0]

            # Check if this message type is supported
            if msg_type not in MESSAGE_DEFINITIONS:
                return []

            # Check if we have enough data
            if len(msg_data) < msg_length:
                return []

            msg_def = MESSAGE_DEFINITIONS[msg_type]
            payload_size = msg_def["payload_size"]

            # Extract and decrypt payload (starts at byte 10)
            if len(msg_data) < 10 + payload_size:
                return []

            encrypted_payload = msg_data[10 : 10 + payload_size]
            decrypted = bytes(b ^ key for b in encrypted_payload)

            chk = calc_checksum(msg_data, 0, len(msg_data) - 2)

            if chk != struct.unpack("<H", msg_data[-2:])[0]:
                logger.info(
                    f"Checksum calculated {chk}, Checksum found: {struct.unpack('<H', msg_data[-2:])[0]}"
                )

                return []

            # Decode the message
            decoded = self._decode_message_data(decrypted, msg_type, tick_val, msg_def)

            return [decoded] if decoded else []

        except (IndexError, struct.error) as e:  # truncated message
            logger.debug(f"Failed to parse message: {e}")
            return []

    def _decode_message_data(
        self,
        decrypted_payload: bytes,
        msg_type: int,
        tick_val: int,
        msg_def: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Unified message decoder using message definition template.

        Decodes message by unpacking each field from its byte offset
        using the specified format code.

        Parameters
        ----------
        decrypted_payload : bytes
            The decrypted message payload.
        msg_type : int
            The message type identifier.
        tick_val : int
            The timestamp value in ticks.
        msg_def : Dict[str, Any]
            Message definition containing field mappings with offsets and format codes.

        Returns
        -------
        Optional[Dict[str, Any]]
            Decoded message as dictionary, or None if decoding fails.
        """
        try:
            result = {"msg_type": msg_type}

            # Unpack each field from its specified offset and format
            for field_info in msg_def["fields"]:
                field_name = msg_def["name"] + ":" + field_info[0]
                fmt_char = field_info[1]  # Format character (e.g., 'i', 'f', 'I')
                offset = field_info[2]  # Byte offset

                # Determine format string for struct.unpack_from
                fmt = f"<{fmt_char}"  # Little-endian

                # Unpack value from offset
                value = struct.unpack_from(fmt, decrypted_payload, offset)[0]

                # Apply conversion function if provided
                if len(field_info) > 3 and callable(field_info[3]):
                    result[field_name] = field_info[3](value)
                else:
                    result[field_name] = value

            # Add tick and timestamp - tick is the reliable time reference
            result["tick"] = tick_val
            # Convert to seconds

            # Try to format datetime from date/time fields (often empty for RTK)
            formatted_dt = self._format_date_time(
                result.get(msg_def["name"] + ":date", 0),
                result.get(msg_def["name"] + ":time", 0),
            )
            if formatted_dt:
                result["datetime"] = formatted_dt  # type: ignore

                fmt = "%Y-%m-%d %H:%M:%S"
                # Parse as UTC to avoid local timezone shifts
                dt = datetime.datetime.strptime(formatted_dt, fmt).replace(
                    tzinfo=datetime.UTC
                )
                result["timestamp"] = dt.timestamp()  # type: ignore

            return result

        except (struct.error, ValueError) as e:  # short payload, invalid date
            logger.error(f"Failed to decode message: {e}")
            return None

    @staticmethod
    def _format_date_time(date: int, time: int) -> str | None:
        """Convert date and time fields into a human-readable datetime string.

        Parameters
        ----------
        date : int
            Date as integer (YYYYMMDD format).
        time : int
            Time as integer (HHMMSS format).

        Returns
        -------
        Optional[str]
            Formatted datetime string, or None if date/time are zero or invalid.
        """
        try:
            # Return None if date/time are zero or invalid
            if date == 0 or time == 0:
                return None

            year = date // 10000
            month = (date % 10000) // 100
            day = date % 100
            hour = time // 10000
            minute = (time % 10000) // 100
            second = time % 100

            # Basic validation
            if year < 2000 or month < 1 or month > 12 or day < 1 or day > 31:
                return None

            return (
                f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            )
        except TypeError as e:  # date or time not an integer
            logger.debug(f"Failed to format datetime: {e}")
            return None

    def _parse_gps_datetime(self, payload: bytes) -> datetime.datetime | None:
        """Parse GPS datetime from message payload.

        Parameters
        ----------
        payload : bytes
            Binary payload containing GPS date and time fields.

        Returns
        -------
        Optional[datetime.datetime]
            Parsed datetime object, or None if parsing fails.
        """
        if len(payload) < 8:
            return None

        raw_date, raw_time = struct.unpack("<II", payload[:8])
        if raw_date == 0:
            return None

        try:
            y = raw_date // 10000
            m = (raw_date % 10000) // 100
            d = raw_date % 100
            H = raw_time // 10000
            M = (raw_time % 10000) // 100
            S = raw_time % 100
            return datetime.datetime(y, m, d, H, M, S, tzinfo=datetime.UTC)
        except ValueError:
            return None
