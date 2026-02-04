## Phase 4 Complete: Fix sensors (IMU, gps, camera)

Added comprehensive type hints, docstrings, and pathlib migration to sensor modules. All three sensor modules now have 100%, 65%, and 74% test coverage respectively with 40 tests passing.

**Files created/changed:**
- tests/test_sensors_imu.py (NEW - 178 lines, 13 tests)
- tests/test_sensors_gps.py (NEW - 156 lines, 10 tests)
- tests/test_sensors_camera.py (NEW - 246 lines, 17 tests)
- pils/sensors/IMU.py
- pils/sensors/gps.py
- pils/sensors/camera.py

**Functions created/changed:**

### IMU.py
- `IMUSensor.__init__(path: str | Path, type: str) -> None` - Added type hints
- `IMUSensor.load_data() -> None` - Added return type, comprehensive docstring
- `IMU.__init__(dirpath: Path) -> None` - Added return type, comprehensive docstring
- `IMU.load_all() -> None` - Added return type, comprehensive docstring

### gps.py
- `GPS.__init__(path: Path, logpath: Optional[Path] = None) -> None` - Added type hints
- `GPS.load_data(freq_interpolation: Optional[float] = None) -> None` - Added type hints, improved docstring
- `GPS._merge_nav_dataframes(...) -> Optional[pl.DataFrame]` - Added return type hint, improved docstring

### camera.py (PATHLIB MIGRATION COMPLETE)
- Removed `import glob` and `import os`
- `Camera.__init__(path: str | Path, logpath: Optional[str | Path] = None, time_index: Optional[Dict[str, Any]] = None) -> None` - Added type hints
- `Camera.load_data() -> None` - Added comprehensive docstring, migrated to pathlib
  - Replaced `glob.glob(os.path.join(self.path, "*.*"))` with `Path(self.path).glob("*.*")`
  - Replaced `os.path.basename(x)` with `Path(x).name` (3 occurrences)
- `Camera.get_frame(frame_number: int) -> np.ndarray` - Added type hints
- `Camera.get_timestamp(frame_number: int) -> Optional[Any]` - Added type hints
- `Camera.plot_frame(frame_number: int, color: str = "rgb") -> None` - Added type hints

**Tests created/changed:**

### test_sensors_imu.py (13 tests):
- `TestIMUSensor` (6 tests): initialization, load_data, column validation, timestamp conversion
- `TestIMU` (7 tests): initialization, load_all, sensor data validation for all 4 sensors

### test_sensors_gps.py (10 tests):
- `TestGPS` (10 tests): initialization, gps.bin detection, load_data, merge_nav_dataframes, type validation

### test_sensors_camera.py (17 tests):
- `TestCamera` (17 tests): initialization, video/image loading, pathlib usage verification, get_frame, get_timestamp

**Coverage improvements:**
- IMU.py: 48% → 100% coverage
- gps.py: 12% → 65% coverage
- camera.py: 17% → 74% coverage

**Review Status:** APPROVED

**Git Commit Message:**
```
feat: add type hints and pathlib to sensors

- Added comprehensive type hints to IMU, GPS, Camera modules
- Added Google-style docstrings with Args/Returns/Raises
- Migrated camera.py to pathlib (removed os.path usage)
- Replaced glob.glob + os.path.join with Path().glob()
- Replaced os.path.basename() with Path().name
- Created 40 comprehensive tests (13 IMU, 10 GPS, 17 Camera)
- All tests passing, coverage: IMU 100%, GPS 65%, Camera 74%
```
