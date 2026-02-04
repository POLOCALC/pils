## Phase 5 Complete: Fix sensors (adc, inclinometer)

Fixed print statements, bare except clauses, and added type hints to ADC and inclinometer sensor modules with comprehensive test coverage.

**Files created/changed:**
- [pils/sensors/adc.py](pils/sensors/adc.py)
- [pils/sensors/inclinometer.py](pils/sensors/inclinometer.py)
- [tests/test_sensors_adc.py](tests/test_sensors_adc.py) (NEW)
- [tests/test_sensors_inclinometer.py](tests/test_sensors_inclinometer.py) (NEW)

**Functions created/changed:**
- `decode_adc_file_struct(adc_path: str | Path) -> pl.DataFrame` - Added type hints
- `decode_adc_file_ascii(adc_path: str | Path, gain_config: int = 16) -> pl.DataFrame` - Added type hints, replaced print() with logging
- `ADC.__init__(path: Path, logpath: Optional[str] = None, gain_config: Optional[int] = None) -> None` - Added type hints
- `ADC.load_data() -> None` - Added type hints and docstring
- `ADC.plot() -> None` - Added type hints and docstring
- `decode_inclino(inclino_path: str | Path) -> Dict[str, List[Any]]` - Added type hints, fixed bare except
- `detect_inclinometer_type_from_config(dirpath: Path) -> Optional[str]` - Fixed bare except
- `KernelInclinometer.__init__(path: Path, logpath: Optional[str] = None) -> None` - Added type hints
- `KernelInclinometer.read_log_time(logfile: Optional[str] = None) -> None` - Added type hints, fixed bare except, replaced print() with logging
- `KernelInclinometer.load_data() -> None` - Added type hints and improved docstring
- `IMX5Inclinometer.__init__(dirpath: Path, logpath: Optional[str] = None) -> None` - Added type hints
- `IMX5Inclinometer.load_ins() -> None` - Added type hints and improved docstring
- `IMX5Inclinometer.load_imu() -> None` - Added type hints and improved docstring
- `Inclinometer.__init__(path: Path, logpath: Optional[str] = None, sensor_type: Optional[Literal["kernel", "imx5"]] = None) -> None` - Added type hints, replaced print() with logging
- `Inclinometer._auto_detect() -> None` - Added type hints
- `Inclinometer._init_decoder() -> None` - Added type hints
- `Inclinometer.load_data() -> None` - Added type hints and improved docstring
- `Inclinometer.plot() -> None` - Added type hints and improved docstring

**Tests created/changed:**
- `TestDecodeAdcFileStruct` (4 tests) - Binary format decoding, type hint validation
- `TestDecodeAdcFileAscii` (7 tests) - ASCII format decoding, gain conversion, logging verification
- `TestADCClass` (9 tests) - ADC initialization, load_data, plot, config auto-detection
- `TestDecodeInclino` (4 tests) - Binary decoding, exception handling, logging verification
- `TestDetectInclinometerTypeFromConfig` (4 tests) - Config-based type detection
- `TestDetectInclinometerTypeFromFiles` (3 tests) - File pattern detection
- `TestKernelInclinometer` (4 tests) - Kernel-100 decoder, log time reading
- `TestIMX5Inclinometer` (4 tests) - IMX-5 CSV decoder, INS/IMU data loading
- `TestInclinometerClass` (9 tests) - Unified inclinometer interface, auto-detection, plotting

**Test Coverage:**
- adc.py: 18% → 91% (111 stmts, 10 miss)
- inclinometer.py: 17% → 83% (256 stmts, 44 miss)
- Total tests: 48/48 passing in 1.74s
- Project coverage: 18% → 25%

**Review Status:** APPROVED

**Git Commit Message:**
```
refactor: fix adc and inclinometer sensors with type hints and logging

- Add type hints to all functions and methods in adc.py and inclinometer.py
- Replace print() statements with centralized logging (logger.warning/error/info)
- Fix bare except clauses to catch Exception with logging
- Create 48 comprehensive tests (20 ADC + 28 inclinometer)
- Improve coverage: adc.py 18%→91%, inclinometer.py 17%→83%
- Add Google-style docstrings with Args/Returns/Raises sections
