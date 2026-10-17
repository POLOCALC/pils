# Inclinometer

::: pils.sensors.inclinometer.Inclinometer

## File formats

| Type | Files | Clock |
|------|-------|-------|
| `imx5` | `*_INC_ins.csv`, `*_INC_imu.csv`, `*_INC_inl2.csv` | payload computer |
| `kernel` | `*_INC.bin` (Kernel-100 messages) | log start + sample counter |
| `legacy` | `*_INC*.pck` (2022; the 2023 files are empty) | payload computer |

CSV lines cut by a power loss (an incomplete last line, NUL bytes) are
dropped.

::: pils.sensors.inclinometer.read_csv_tolerant
