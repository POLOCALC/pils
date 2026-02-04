## Phase 3 Complete: Fix loader/stout.py

Replaced all `os` and `os.path` usage with `pathlib` in `pils/loader/stout.py`. All 20+ occurrences migrated successfully. Created comprehensive test suite with 18 tests covering filesystem fallback functionality.

**Files created/changed:**
- tests/test_stout_loader.py (NEW)
- pils/loader/stout.py

**Functions created/changed:**
- `StoutLoader._load_all_flights_from_filesystem()` - Replaced `os.listdir()` with `.iterdir()`, `os.path.join()` with `Path() / "subdir"`
- `StoutLoader._build_flight_dict_from_filesystem()` - Uses `Path()` for path construction
- `StoutLoader._list_files_recursive()` - Replaced `os.walk()` with `Path.rglob()`
- `StoutLoader._get_data_files()` - Uses `Path().exists()` instead of `os.path.exists()`
- `StoutLoader._get_campaigns_from_filesystem()` - Uses `.iterdir()` and `.is_dir()`
- `StoutLoader._load_sensor_dataframe()` - Uses `Path().exists()` for validation
- `StoutLoader._load_drone_dataframe()` - Uses `Path().exists()` for validation

**Tests created/changed:**
- `TestStoutLoaderInitialization::test_init_attempts_stout_import` - Verify STOUT import fallback
- `TestLoadAllFlightsFromFilesystem` (5 tests) - Test loading all flights from filesystem
- `TestLoadSingleFlightFromFilesystem` (2 tests) - Test loading single flight by name
- `TestBuildFlightDictFromFilesystem` (2 tests) - Test flight dictionary construction
- `TestListFilesRecursive` (2 tests) - Test recursive file listing with pathlib
- `TestGetCampaignListFromFilesystem` (2 tests) - Test campaign listing
- `TestPathlibUsage` (2 tests) - Verify pathlib is used correctly
- `TestLoadFlightsByDateFromFilesystem` (2 tests) - Test date filtering
- `TestCollectSpecificData` (1 test) - Test data collection

**Review Status:** APPROVED

**Git Commit Message:**
```
refactor: migrate loader/stout.py to pathlib

- Replaced 20+ os.path occurrences with pathlib equivalents
- Updated _load_all_flights_from_filesystem to use Path.iterdir()
- Updated _list_files_recursive to use Path.rglob()
- Updated _get_campaigns_from_filesystem to use Path methods
- Added 18 comprehensive tests with STOUT mocking
- All tests passing with proper filesystem fallback coverage
```
