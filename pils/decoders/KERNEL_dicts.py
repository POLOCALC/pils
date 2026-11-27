"""Kernel-100 inclinometer message tables and scale factors."""

g = 9.8106
GYRO_SCALE = 1e5
ACC_SCALE = 1e6 * g

WORD = "H"
SWORD = "h"
WORD_LONG = "I"
SWORD_LONG = "i"


USW_TABLE = {
    "low": [
        "Reserved",
        "Internal Comms",
        "Internal Config",
        "Reserved",
        "Reserved",
        "Reserved",
        "Reserved",
        "Reserved",
    ],
    "high": [
        "Reserved",
        "Reserved",
        "AngRateX",
        "AngRateX",
        "AngRateX",
        "Reserved",
        "Temperature",
        "Reserved",
    ],
}


def extract_USW(USW):
    """Decode the Kernel-100 unit status word into OK/KO flags."""

    def check_byte(byte):
        if isinstance(byte, int):
            return format(byte, "08b")  # Ensure 8-bit representation
        elif isinstance(byte, str):
            return format(int(byte, base=16), "08b")  # Convert hex to binary

    def check_values(bits, table):
        result = []
        for i, bit in enumerate(reversed(bits)):  # Reverse bits (LSB first)
            if table[i] == "Reserved":
                result.append(f"RES_{i}")
            else:
                result.append(f"KO_{table[i]}" if bit == "1" else f"OK_{table[i]}")

        return ", ".join(result)  # Return a readable string

    low = check_byte(USW[0])
    high = check_byte(USW[1])

    return check_values(low, USW_TABLE["low"]), check_values(high, USW_TABLE["high"])


MODES = {
    "KERNEL_Orientation": {
        "Address": b"\x33",
        "length": 34,
        "Parameters": [
            "Heading",
            "Pitch",
            "Roll",
            "GyroX",
            "GyroY",
            "GyroZ",
            "AccX",
            "AccY",
            "AccZ",
            "MagX",
            "MagY",
            "MagZ",
            "Reserved",
            "USW",
            "Vinp",
            "Temper",
        ],
        "Type": [
            WORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            WORD_LONG,
            WORD,
            WORD,
            SWORD,
        ],
        "Scale": [
            100,
            100,
            100,
            10,
            10,
            10,
            2000 * g,
            2000 * g,
            2000 * g,
            1,
            1,
            1,
            1,
            1,
            100,
            10,
        ],
    },
    "KERNEL_GAData": {
        "Address": b"\x8f",
        "Length": 32,
        "Parameters": [
            "GyroX",
            "GyroY",
            "GyroZ",
            "AccX",
            "AccY",
            "AccZ",
            "Reserved",
            "USW",
            "Vinp",
            "Temper",
        ],
        "Type": [
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            WORD,
            WORD,
            WORD,
            SWORD,
        ],
        "Scale": [1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1, 1, 100, 10],
    },
    "KERNEL_GAmData": {
        "Address": b"\x9b",
        "Length": 32,
        "Parameters": [
            "GyroX",
            "GyroY",
            "GyroZ",
            "AccX",
            "AccY",
            "AccZ",
            "Time Flag",
            "Second Fraction",
            "Temper",
        ],
        "Type": [
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            WORD,
            WORD_LONG,
            SWORD,
        ],
        "Scale": [1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1, 1, 10],
    },
    "KERNEL_QuatData": {
        "Address": b"\x82",
        "length": 34,
        "Parameters": [
            "Heading",
            "Pitch",
            "Roll",
            "q0",
            "q1",
            "q2",
            "q3",
            "Reserved",
            "USW",
            "Vinp",
            "Temper",
        ],
        "Type": [
            WORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            WORD_LONG * 3 + WORD,
            WORD,
            WORD,
            SWORD,
        ],
        "Scale": [100, 100, 100, 10000, 10000, 10000, 10000, 1, 1, 100, 10],
    },
    "KERNEL_CalibHR": {
        "Address": b"\x81",
        "length": 52,
        "Parameters": [
            "Heading",
            "Pitch",
            "Roll",
            "GyroX",
            "GyroY",
            "GyroZ",
            "AccX",
            "AccY",
            "AccZ",
            "MagX",
            "MagY",
            "MagZ",
            "Counter",
            "Reserved",
            "USW",
            "Vinp",
            "Temper",
        ],
        "Type": [
            WORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            SWORD,
            WORD,
            WORD,
            SWORD,
        ],
        "Scale": [
            1000,
            1000,
            1000,
            GYRO_SCALE,
            GYRO_SCALE,
            GYRO_SCALE,
            ACC_SCALE,
            ACC_SCALE,
            ACC_SCALE,
            1,
            1,
            1,
            1,
            1,
            1,
            100,
            10,
        ],
    },
    ### ONLY FOR KERNEL-120 and KERNEL-220
    "KERNEL_GAAmData": {
        "Address": b"\xa5",
        "Length": 44,
        "Parameters": [
            "GyroX",
            "GyroY",
            "GyroZ",
            "Acc1X",
            "Acc1Y",
            "Acc1Z",
            "Acc2X",
            "Acc2Y",
            "Acc2Z",
            "Reserved",
            "USW",
            "Vinp",
            "Temper",
        ],
        "Type": [
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            WORD,
            WORD,
            WORD,
            SWORD,
        ],
        "Scale": [1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1, 1, 100, 10],
    },
    "KERNEL_GAAData": {
        "Address": b"\xa6",
        "Length": 44,
        "Parameters": [
            "GyroX",
            "GyroY",
            "GyroZ",
            "Acc1X",
            "Acc1Y",
            "Acc1Z",
            "Acc2X",
            "Acc2Y",
            "Acc2Z",
            "Time Flag",
            "Second Fraction",
            "Temper",
        ],
        "Type": [
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            SWORD_LONG,
            WORD,
            WORD_LONG,
            SWORD,
        ],
        "Scale": [1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1, 1, 10],
    },
    "USER_DEFINED_DATA": {"Address": b"\x95"},
    "USER_DEFINED_DATA_CONFIG": {"Address": b"\x96"},
    "STOP": {"Address": b"\xfe"},
}
