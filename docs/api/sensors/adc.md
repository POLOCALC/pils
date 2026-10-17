# ADC

::: pils.sensors.adc.ADC

## File formats

- ASCII: time (µs) and ADS1015 counts per line (converted to mV with the
  gain of `config.yml`).
- Binary, 20-byte records: time (s), reading time (µs), amplitude (mV).
- Binary, 12-byte records (April 2024): time (s), amplitude (V, converted
  to mV).
- 2022 and 2023: pickle files, see below.

The binary layout is detected from the data; a record cut at the end and
preallocated zeros are dropped.

::: pils.sensors.adc.detect_struct_layout

## 2022 and 2023 payloads

::: pils.sensors.legacy
    options:
      members:
        - read_legacy_adc
        - read_legacy_inclinometer
        - read_pickles
        - undo_clock_jumps
        - adc_blocks_to_frame
