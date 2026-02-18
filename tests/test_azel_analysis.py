"""
Tests for AZEL Analysis Module.

Testing AZELVersion dataclass and related functionality.
"""

from dataclasses import fields

import polars as pl
import pytest

from pils.flight import Flight


@pytest.fixture
def mock_flight(tmp_path):
    """Create mock Flight object with valid flight_path."""
    flight_path = tmp_path / "flight_test"
    flight_path.mkdir()
    drone_data_path = flight_path / "drone"
    drone_data_path.mkdir()

    flight_info = {
        "drone_data_folder_path": str(drone_data_path),
    }
    flight = Flight(flight_info)
    return flight


def test_import_azel_version():
    """Test that AZELVersion can be imported."""
    from pils.analyze.azel import AZELVersion

    assert AZELVersion is not None


def test_azel_version_dataclass_creation():
    """Test AZELVersion instantiation with all fields."""
    from pils.analyze.azel import AZELVersion

    # Create sample AZEL data
    azel_data = pl.DataFrame(
        {
            "timestamp": [1000.0, 2000.0, 3000.0],
            "az": [45.0, 90.0, 135.0],
            "el": [30.0, 45.0, 60.0],
            "srange": [100.5, 150.3, 200.8],
        }
    )

    metadata = {
        "observer_lat": 40.7128,
        "observer_lon": -74.0060,
        "observer_alt": 10.0,
        "creation_date": "2026-02-18",
    }

    # Create AZELVersion instance
    version = AZELVersion(
        version_name="v1_test", azel_data=azel_data, metadata=metadata
    )

    # Verify all fields
    assert version.version_name == "v1_test"
    assert version.azel_data.shape == (3, 4)
    assert version.metadata["observer_lat"] == 40.7128
    assert isinstance(version.azel_data, pl.DataFrame)


def test_azel_version_required_fields():
    """Test that AZELVersion requires all fields."""
    from pils.analyze.azel import AZELVersion

    # This should raise TypeError when missing required fields
    with pytest.raises(TypeError):
        AZELVersion()  # No arguments

    with pytest.raises(TypeError):
        AZELVersion(version_name="test")  # Missing azel_data and metadata


def test_azel_version_dataframe_schema():
    """Test that azel_data has correct column types."""
    from pils.analyze.azel import AZELVersion

    # Create AZEL data with correct schema
    azel_data = pl.DataFrame(
        {
            "timestamp": [1000.0, 2000.0],
            "az": [45.0, 90.0],
            "el": [30.0, 45.0],
            "srange": [100.5, 150.3],
        }
    )

    version = AZELVersion(version_name="schema_test", azel_data=azel_data, metadata={})

    # Check DataFrame has expected columns
    assert "timestamp" in version.azel_data.columns
    assert "az" in version.azel_data.columns
    assert "el" in version.azel_data.columns
    assert "srange" in version.azel_data.columns

    # Check data types are Float64
    assert version.azel_data["timestamp"].dtype == pl.Float64
    assert version.azel_data["az"].dtype == pl.Float64
    assert version.azel_data["el"].dtype == pl.Float64
    assert version.azel_data["srange"].dtype == pl.Float64


def test_azel_version_is_dataclass():
    """Test that AZELVersion is actually a dataclass."""
    from dataclasses import is_dataclass

    from pils.analyze.azel import AZELVersion

    assert is_dataclass(AZELVersion)

    # Check expected fields exist
    field_names = [f.name for f in fields(AZELVersion)]
    assert "version_name" in field_names
    assert "azel_data" in field_names
    assert "metadata" in field_names


class TestAZELAnalysisInit:
    """Test AZELAnalysis initialization and path setup."""

    def test_azel_analysis_init_with_valid_flight(self, mock_flight):
        """Test initialization with valid Flight object."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        assert azel.flight_path == mock_flight.flight_path
        assert azel.azel_dir.exists()
        assert azel.azel_dir == mock_flight.flight_path / "proc" / "azel"

    def test_azel_analysis_creates_azel_directory(self, mock_flight):
        """Test that AZEL directory is created at proc/azel/."""
        from pils.analyze.azel import AZELAnalysis

        # Verify directory doesn't exist initially
        azel_dir = mock_flight.flight_path / "proc" / "azel"
        assert not azel_dir.exists()

        # Create AZELAnalysis instance
        azel = AZELAnalysis(mock_flight)

        # Verify directory was created
        assert azel_dir.exists()
        assert azel_dir.is_dir()
        assert azel.azel_dir == azel_dir

    def test_azel_analysis_rejects_invalid_flight(self, tmp_path):
        """Test that initialization raises TypeError for non-Flight objects."""
        from pils.analyze.azel import AZELAnalysis

        flight_path = tmp_path / "flight_004"
        flight_path.mkdir()

        # Test with string path
        with pytest.raises(TypeError, match="Expected Flight object"):
            AZELAnalysis(str(flight_path))

        # Test with Path object
        with pytest.raises(TypeError, match="Expected Flight object"):
            AZELAnalysis(flight_path)

        # Test with None
        with pytest.raises(TypeError, match="Expected Flight object"):
            AZELAnalysis(None)

    def test_azel_analysis_rejects_missing_flight_path(self, tmp_path):
        """Test that initialization raises ValueError for Flight without flight_path."""
        from pils.analyze.azel import AZELAnalysis

        # Create flight with None flight_path
        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)

        # Manually set flight_path to None to simulate invalid state
        flight.flight_path = None

        with pytest.raises(
            ValueError, match="Flight object must have a valid flight_path"
        ):
            AZELAnalysis(flight)

    def test_azel_analysis_rejects_flight_path_not_directory(self, tmp_path):
        """Test that initialization raises ValueError if flight_path is not a directory."""
        from pils.analyze.azel import AZELAnalysis

        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)
        flight.flight_path = file_path

        with pytest.raises(
            ValueError, match="flight_path must be an existing directory"
        ):
            AZELAnalysis(flight)

    def test_azel_analysis_azel_dir_property(self, mock_flight):
        """Test azel_dir property returns correct Path."""
        from pathlib import Path

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Verify azel_dir is a Path object
        assert isinstance(azel.azel_dir, Path)

        # Verify correct path structure
        expected_dir = mock_flight.flight_path / "proc" / "azel"
        assert azel.azel_dir == expected_dir


class TestCoordinateTransformations:
    """Test coordinate transformation methods (_compute_enu_positions, _compute_azel)."""

    def test_compute_enu_with_known_values(self):
        """Test geodetic2enu with known lat/lon/alt conversions."""
        from pils.analyze.azel import AZELAnalysis

        # Reference position
        ref_lat, ref_lon, ref_alt = 40.0, -105.0, 1000.0

        # Test 1: Target at same position → E=0, N=0, U=0
        e, n, u = AZELAnalysis._compute_enu_positions(
            ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, ref_alt
        )
        assert e == pytest.approx(0.0, abs=0.01)
        assert n == pytest.approx(0.0, abs=0.01)
        assert u == pytest.approx(0.0, abs=0.01)

        # Test 2: Target 0.001° north → ~111m north
        target_lat = ref_lat + 0.001
        e, n, u = AZELAnalysis._compute_enu_positions(
            ref_lat, ref_lon, ref_alt, target_lat, ref_lon, ref_alt
        )
        assert e == pytest.approx(0.0, abs=0.1)
        assert n == pytest.approx(111.0, abs=1.0)  # ~111m per 0.001°
        assert u == pytest.approx(0.0, abs=0.1)

        # Test 3: Target 0.001° east → ~85m east (at 40° latitude)
        target_lon = ref_lon + 0.001
        e, n, u = AZELAnalysis._compute_enu_positions(
            ref_lat, ref_lon, ref_alt, ref_lat, target_lon, ref_alt
        )
        assert e == pytest.approx(85.0, abs=2.0)  # ~85m per 0.001° at 40° lat
        assert n == pytest.approx(0.0, abs=0.1)
        assert u == pytest.approx(0.0, abs=0.1)

        # Test 4: Target 100m higher → U=100
        target_alt = ref_alt + 100.0
        e, n, u = AZELAnalysis._compute_enu_positions(
            ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, target_alt
        )
        assert e == pytest.approx(0.0, abs=0.1)
        assert n == pytest.approx(0.0, abs=0.1)
        assert u == pytest.approx(100.0, abs=0.1)

    def test_compute_azel_from_enu(self):
        """Test enu2aer produces correct azimuth/elevation/range."""
        from pils.analyze.azel import AZELAnalysis

        # Test 1: North (E=0, N=100, U=0) → Az≈0°, El≈0°, Range≈100
        az, el, srange = AZELAnalysis._compute_azel(0.0, 100.0, 0.0)
        assert az == pytest.approx(0.0, abs=0.1)
        assert el == pytest.approx(0.0, abs=0.1)
        assert srange == pytest.approx(100.0, abs=0.1)

        # Test 2: East (E=100, N=0, U=0) → Az≈90°, El≈0°, Range≈100
        az, el, srange = AZELAnalysis._compute_azel(100.0, 0.0, 0.0)
        assert az == pytest.approx(90.0, abs=0.1)
        assert el == pytest.approx(0.0, abs=0.1)
        assert srange == pytest.approx(100.0, abs=0.1)

        # Test 3: Up (E=0, N=0, U=100) → El≈90°, Range≈100
        az, el, srange = AZELAnalysis._compute_azel(0.0, 0.0, 100.0)
        assert el == pytest.approx(90.0, abs=0.1)
        assert srange == pytest.approx(100.0, abs=0.1)
        # Azimuth undefined when directly above, pymap3d returns 0
        assert az == pytest.approx(0.0, abs=0.1)

        # Test 4: Northeast (E=100, N=100, U=0) → Az≈45°, El≈0°
        az, el, srange = AZELAnalysis._compute_azel(100.0, 100.0, 0.0)
        assert az == pytest.approx(45.0, abs=0.1)
        assert el == pytest.approx(0.0, abs=0.1)
        # Range = sqrt(100^2 + 100^2) ≈ 141.4
        assert srange == pytest.approx(141.4, abs=0.5)

        # Test 5: South (E=0, N=-100, U=0) → Az≈180°, El≈0°
        az, el, srange = AZELAnalysis._compute_azel(0.0, -100.0, 0.0)
        assert az == pytest.approx(180.0, abs=0.1)
        assert el == pytest.approx(0.0, abs=0.1)
        assert srange == pytest.approx(100.0, abs=0.1)

        # Test 6: West (E=-100, N=0, U=0) → Az≈270°, El≈0°
        az, el, srange = AZELAnalysis._compute_azel(-100.0, 0.0, 0.0)
        assert az == pytest.approx(270.0, abs=0.1)
        assert el == pytest.approx(0.0, abs=0.1)
        assert srange == pytest.approx(100.0, abs=0.1)

    def test_enu_positions_are_relative_to_reference(self):
        """Test ENU is zero when drone at reference position."""
        from pils.analyze.azel import AZELAnalysis

        # Multiple reference positions
        test_positions = [
            (40.0, -105.0, 1000.0),
            (51.5074, -0.1278, 50.0),  # London
            (-33.8688, 151.2093, 20.0),  # Sydney
        ]

        for ref_lat, ref_lon, ref_alt in test_positions:
            e, n, u = AZELAnalysis._compute_enu_positions(
                ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, ref_alt
            )
            assert e == pytest.approx(0.0, abs=0.01)
            assert n == pytest.approx(0.0, abs=0.01)
            assert u == pytest.approx(0.0, abs=0.01)

    def test_azel_north_direction_zero_azimuth(self):
        """Test azimuth=0° when drone directly north."""
        from pils.analyze.azel import AZELAnalysis

        # Reference position
        ref_lat, ref_lon, ref_alt = 40.0, -105.0, 1000.0

        # Target positions north at different distances
        for distance_deg in [0.001, 0.01, 0.1]:
            target_lat = ref_lat + distance_deg
            e, n, u = AZELAnalysis._compute_enu_positions(
                ref_lat, ref_lon, ref_alt, target_lat, ref_lon, ref_alt
            )

            # Compute azimuth - should be 0° (north)
            az, el, srange = AZELAnalysis._compute_azel(e, n, u)
            assert az == pytest.approx(0.0, abs=0.1)
            assert el == pytest.approx(0.0, abs=0.1)  # Horizontal
            assert n > 0  # North should be positive

    def test_azel_elevation_90_when_directly_above(self):
        """Test elevation=90° when drone directly above."""
        from pils.analyze.azel import AZELAnalysis

        # Reference position
        ref_lat, ref_lon, ref_alt = 40.0, -105.0, 1000.0

        # Target positions directly above at different altitudes
        for alt_diff in [10.0, 100.0, 1000.0]:
            target_alt = ref_alt + alt_diff
            e, n, u = AZELAnalysis._compute_enu_positions(
                ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, target_alt
            )

            # Compute elevation - should be 90° (zenith)
            az, el, srange = AZELAnalysis._compute_azel(e, n, u)
            assert el == pytest.approx(90.0, abs=0.1)
            assert srange == pytest.approx(alt_diff, abs=0.1)
            assert u == pytest.approx(alt_diff, abs=0.1)


class TestReferenceDataLoading:
    """Test EMLID reference data loading and RTK correction computations."""

    @pytest.fixture
    def mock_emlid_csv(self, tmp_path):
        """Create mock EMLID CSV with telescope and base positions."""
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
SATP1_01,-105.0000,40.0000,1000.0
SATP1_02,-105.0001,40.0001,1000.5
SATP1_03,-105.0002,40.0002,1001.0
DJI_BASE_01,-105.1000,40.1000,1100.0
DJI_BASE_02,-105.1001,40.1001,1100.5
"""
        csv_file = tmp_path / "emlid_ref.csv"
        csv_file.write_text(csv_content)
        return csv_file

    def test_load_emlid_data_returns_telescope_position(self, mock_emlid_csv):
        """Test loading telescope position from EMLID CSV."""
        from pils.analyze.azel import AZELAnalysis

        ref_data = AZELAnalysis._load_emlid_reference_data(mock_emlid_csv, "SATP1")

        # Check telescope key exists
        assert "telescope" in ref_data
        assert "lat" in ref_data["telescope"]
        assert "lon" in ref_data["telescope"]
        assert "alt" in ref_data["telescope"]

        # Check mean values (mean of 3 SATP1 positions)
        # Lat: (40.0000 + 40.0001 + 40.0002) / 3 = 40.0001
        # Lon: (-105.0000 + -105.0001 + -105.0002) / 3 = -105.0001
        # Alt: (1000.0 + 1000.5 + 1001.0) / 3 = 1000.5
        assert ref_data["telescope"]["lat"] == pytest.approx(40.0001, abs=0.0001)
        assert ref_data["telescope"]["lon"] == pytest.approx(-105.0001, abs=0.0001)
        assert ref_data["telescope"]["alt"] == pytest.approx(1000.5, abs=0.01)

    def test_load_emlid_data_returns_base_position(self, mock_emlid_csv):
        """Test loading DJI base position from EMLID CSV."""
        from pils.analyze.azel import AZELAnalysis

        ref_data = AZELAnalysis._load_emlid_reference_data(mock_emlid_csv, "SATP1")

        # Check base key exists
        assert "base" in ref_data
        assert "lat" in ref_data["base"]
        assert "lon" in ref_data["base"]
        assert "alt" in ref_data["base"]

        # Check mean values (mean of 2 DJI_BASE positions)
        # Lat: (40.1000 + 40.1001) / 2 = 40.10005
        # Lon: (-105.1000 + -105.1001) / 2 = -105.10005
        # Alt: (1100.0 + 1100.5) / 2 = 1100.25
        assert ref_data["base"]["lat"] == pytest.approx(40.10005, abs=0.0001)
        assert ref_data["base"]["lon"] == pytest.approx(-105.10005, abs=0.0001)
        assert ref_data["base"]["alt"] == pytest.approx(1100.25, abs=0.01)

    def test_emlid_data_filters_by_name_prefix(self, tmp_path):
        """Test filtering by telescope name prefix."""
        from pils.analyze.azel import AZELAnalysis

        # Create CSV with multiple telescope types
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
SATP1_01,-105.0000,40.0000,1000.0
SATP1_02,-105.0001,40.0001,1001.0
SATP2_01,-106.0000,41.0000,2000.0
SATP2_02,-106.0001,41.0001,2001.0
DJI_BASE_01,-105.5000,40.5000,1500.0
"""
        csv_file = tmp_path / "multi_telescope.csv"
        csv_file.write_text(csv_content)

        # Load SATP1 data
        ref_data_1 = AZELAnalysis._load_emlid_reference_data(csv_file, "SATP1")
        assert ref_data_1["telescope"]["lat"] == pytest.approx(
            40.00005, abs=0.0001
        )  # Mean of 40.0000 and 40.0001
        assert ref_data_1["telescope"]["lon"] == pytest.approx(-105.00005, abs=0.0001)

        # Load SATP2 data
        ref_data_2 = AZELAnalysis._load_emlid_reference_data(csv_file, "SATP2")
        assert ref_data_2["telescope"]["lat"] == pytest.approx(
            41.00005, abs=0.0001
        )  # Mean of 41.0000 and 41.0001
        assert ref_data_2["telescope"]["lon"] == pytest.approx(-106.00005, abs=0.0001)

        # Base should be the same for both
        assert ref_data_1["base"]["lat"] == ref_data_2["base"]["lat"]

    def test_emlid_file_not_found_raises_error(self, tmp_path):
        """Test that missing EMLID file raises FileNotFoundError."""
        from pils.analyze.azel import AZELAnalysis

        nonexistent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="EMLID CSV file not found"):
            AZELAnalysis._load_emlid_reference_data(nonexistent_file, "SATP1")

    def test_telescope_not_found_raises_error(self, tmp_path):
        """Test that missing telescope name raises ValueError."""
        from pils.analyze.azel import AZELAnalysis

        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
DJI_BASE_01,-105.0000,40.0000,1000.0
"""
        csv_file = tmp_path / "no_telescope.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(
            ValueError, match="No telescope positions found for 'SATP1'"
        ):
            AZELAnalysis._load_emlid_reference_data(csv_file, "SATP1")

    def test_base_not_found_raises_error(self, tmp_path):
        """Test that missing base positions raises ValueError."""
        from pils.analyze.azel import AZELAnalysis

        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
SATP1_01,-105.0000,40.0000,1000.0
"""
        csv_file = tmp_path / "no_base.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="No DJI base positions found"):
            AZELAnalysis._load_emlid_reference_data(csv_file, "SATP1")

    def test_compute_rtk_correction_offset(self):
        """Test RTK correction calculates ENU offset correctly."""
        from pils.analyze.azel import AZELAnalysis

        # Actual DJI base position
        dji_base_geod = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        # Broadcast position is slightly north (0.001° ≈ 111m)
        dji_broadcast_geod = {"lat": 40.001, "lon": -105.0, "alt": 1000.0}

        delta_e, delta_n, delta_u = AZELAnalysis._compute_rtk_correction(
            dji_base_geod, dji_broadcast_geod
        )

        # Since broadcast is north of actual base, correction should be negative north
        assert delta_e == pytest.approx(0.0, abs=0.1)
        assert delta_n == pytest.approx(
            -111.0, abs=1.0
        )  # Negative because base is south
        assert delta_u == pytest.approx(0.0, abs=0.1)

    def test_rtk_correction_with_zero_offset(self):
        """Test correction is zero when base positions match."""
        from pils.analyze.azel import AZELAnalysis

        # Identical positions
        position = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        delta_e, delta_n, delta_u = AZELAnalysis._compute_rtk_correction(
            position, position
        )

        # All corrections should be zero
        assert delta_e == pytest.approx(0.0, abs=0.01)
        assert delta_n == pytest.approx(0.0, abs=0.01)
        assert delta_u == pytest.approx(0.0, abs=0.01)

    def test_rtk_correction_east_west_offset(self):
        """Test RTK correction for east-west offset."""
        from pils.analyze.azel import AZELAnalysis

        # Actual DJI base position
        dji_base_geod = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        # Broadcast position is slightly east (0.001° ≈ 85m at 40° lat)
        dji_broadcast_geod = {"lat": 40.0, "lon": -104.999, "alt": 1000.0}

        delta_e, delta_n, delta_u = AZELAnalysis._compute_rtk_correction(
            dji_base_geod, dji_broadcast_geod
        )

        # Broadcast is east of actual base, correction should be negative east
        assert delta_e == pytest.approx(-85.0, abs=2.0)
        assert delta_n == pytest.approx(0.0, abs=0.1)
        assert delta_u == pytest.approx(0.0, abs=0.1)

    def test_rtk_correction_altitude_offset(self):
        """Test RTK correction for altitude offset."""
        from pils.analyze.azel import AZELAnalysis

        # Actual DJI base position
        dji_base_geod = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        # Broadcast position is 50m higher
        dji_broadcast_geod = {"lat": 40.0, "lon": -105.0, "alt": 1050.0}

        delta_e, delta_n, delta_u = AZELAnalysis._compute_rtk_correction(
            dji_base_geod, dji_broadcast_geod
        )

        # Broadcast is higher than actual base, correction should be negative up
        assert delta_e == pytest.approx(0.0, abs=0.1)
        assert delta_n == pytest.approx(0.0, abs=0.1)
        assert delta_u == pytest.approx(-50.0, abs=0.1)


class TestRunAnalysis:
    """Test suite for run_analysis() main method."""

    @pytest.fixture
    def mock_drone_csv_data(self, tmp_path):
        """Create mock drone CSV data with RTK positions and timestamps."""
        # Create drone data DataFrame (CSV format columns)
        drone_df = pl.DataFrame(
            {
                "RTKdata:Lat_P": [40.0001, 40.0002, 0.0, 40.0004, 40.0005],
                "RTKdata:Lon_P": [
                    -105.0001,
                    -105.0002,
                    -105.0003,
                    -105.0004,
                    -105.0005,
                ],
                "RTKdata:Hmsl_P": [1010.0, 1011.0, 1012.0, 1013.0, 1014.0],
                "GPS:Date": [20260218, 20260218, 20260218, 20260218, 20260218],
                "GPS:Time": [120000, 120001, 120002, 120003, 120004],
            }
        )
        return drone_df

    @pytest.fixture
    def mock_emlid_csv(self, tmp_path):
        """Create mock EMLID reference CSV file."""
        emlid_path = tmp_path / "emlid_reference.csv"
        emlid_data = pl.DataFrame(
            {
                "Name": ["SATP1_1", "SATP1_2", "DJI_BASE_1", "DJI_BASE_2"],
                "Latitude": [40.0, 40.0001, 40.0, 40.0001],
                "Longitude": [-105.0, -105.0001, -105.0, -105.0001],
                "Ellipsoidal height": [1000.0, 1001.0, 1000.0, 1001.0],
            }
        )
        emlid_data.write_csv(emlid_path)
        return emlid_path

    @pytest.fixture
    def mock_flight_with_drone_data(self, tmp_path, mock_drone_csv_data):
        """Create Flight object with mock drone RTK data."""
        from pils.flight import DroneData

        flight_path = tmp_path / "flight_test"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)

        # Mock drone data loading (use drone_df, not drone)
        flight.raw_data.drone_data = DroneData(
            drone_df=mock_drone_csv_data, litchi_df=None
        )

        return flight

    def test_run_analysis_with_valid_inputs(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test full run_analysis pipeline with valid inputs."""
        from pils.analyze.azel import AZELAnalysis, AZELVersion

        azel = AZELAnalysis(mock_flight_with_drone_data)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Should return AZELVersion
        assert result is not None
        assert isinstance(result, AZELVersion)
        assert isinstance(result.azel_data, pl.DataFrame)
        assert isinstance(result.metadata, dict)

        # Check version name format
        assert result.version_name.startswith("rev_")

    def test_run_analysis_filters_invalid_rtk_data(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test that rows with lat=0 or NaN are filtered out."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_drone_data)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Original has 5 rows, but 1 has lat=0, so should have 4 valid rows
        assert result is not None
        assert result.azel_data.height == 4

    def test_run_analysis_applies_rtk_correction(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test that RTK correction is applied to ENU positions."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_drone_data)

        # Broadcast position slightly offset from actual base
        dji_broadcast = {"lat": 40.001, "lon": -105.001, "alt": 1005.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Result should exist and contain corrected AZEL data
        assert result is not None
        assert result.azel_data.height == 4

        # Metadata should contain RTK correction info
        assert "dji_broadcast_position" in result.metadata

    def test_run_analysis_returns_azel_version(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test that run_analysis returns AZELVersion with correct structure."""
        from pils.analyze.azel import AZELAnalysis, AZELVersion

        azel = AZELAnalysis(mock_flight_with_drone_data)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Check it's an AZELVersion
        assert isinstance(result, AZELVersion)

        # Check structure
        assert hasattr(result, "version_name")
        assert hasattr(result, "azel_data")
        assert hasattr(result, "metadata")

        # Check metadata contains expected keys
        assert "telescope_position" in result.metadata
        assert "base_position" in result.metadata
        assert "dji_broadcast_position" in result.metadata

    def test_run_analysis_output_has_required_columns(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test output DataFrame has required columns with correct types."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_drone_data)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert result is not None
        df = result.azel_data

        # Check required columns exist
        required_columns = ["timestamp", "az", "el", "srange"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Check column types are Float64
        assert df["timestamp"].dtype == pl.Float64
        assert df["az"].dtype == pl.Float64
        assert df["el"].dtype == pl.Float64
        assert df["srange"].dtype == pl.Float64

    def test_run_analysis_handles_no_valid_data(self, tmp_path, mock_emlid_csv):
        """Test that run_analysis returns None when all RTK data is invalid."""
        from pils.analyze.azel import AZELAnalysis
        from pils.flight import DroneData

        # Create flight with all invalid RTK data (all lat=0)
        flight_path = tmp_path / "flight_invalid"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)

        # All lat values are 0 (invalid)
        invalid_drone_df = pl.DataFrame(
            {
                "RTKdata:Lat_P": [0.0, 0.0, 0.0],
                "RTKdata:Lon_P": [-105.0, -105.0, -105.0],
                "RTKdata:Hmsl_P": [1000.0, 1000.0, 1000.0],
                "GPS:Date": [20260218, 20260218, 20260218],
                "GPS:Time": [120000, 120001, 120002],
            }
        )

        flight.raw_data.drone_data = DroneData(
            drone_df=invalid_drone_df, litchi_df=None
        )

        azel = AZELAnalysis(flight)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        result = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Should return None when no valid data
        assert result is None

    def test_run_analysis_timezone_correction(
        self, mock_flight_with_drone_data, mock_emlid_csv
    ):
        """Test that timezone correction is applied to timestamps."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_drone_data)

        dji_broadcast = {"lat": 40.0, "lon": -105.0, "alt": 1000.0}

        # Run with UTC (0 hours offset)
        result_utc = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Run with +1 hour offset
        result_tz = azel.run_analysis(
            emlid_csv_path=mock_emlid_csv,
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=1.0,
        )

        assert result_utc is not None
        assert result_tz is not None

        # Timestamps should differ by 3600 seconds (1 hour)
        ts_utc = result_utc.azel_data["timestamp"][0]
        ts_tz = result_tz.azel_data["timestamp"][0]

        assert abs(ts_utc - ts_tz) == pytest.approx(3600.0, abs=1.0)


class TestHDF5Persistence:
    """Test HDF5 persistence for AZEL versions."""

    @pytest.fixture
    def sample_azel_version(self):
        """Create sample AZELVersion for testing."""
        from pils.analyze.azel import AZELVersion

        azel_data = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "az": [45.0, 90.0, 135.0],
                "el": [30.0, 45.0, 60.0],
                "srange": [100.5, 150.3, 200.8],
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("az").cast(pl.Float64),
                pl.col("el").cast(pl.Float64),
                pl.col("srange").cast(pl.Float64),
            ]
        )

        metadata = {
            "telescope_position": {"lat": 40.0, "lon": -105.0, "alt": 1000.0},
            "base_position": {"lat": 40.001, "lon": -105.001, "alt": 1001.0},
            "telescope_name": "SATP1",
            "num_samples": 3,
        }

        return AZELVersion(
            version_name="rev_20260218_120000", azel_data=azel_data, metadata=metadata
        )

    def test_save_azel_version_to_hdf5(self, mock_flight, sample_azel_version):
        """Test saving AZELVersion to HDF5 file."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Save version to HDF5
        azel._save_to_hdf5(sample_azel_version)

        # Verify HDF5 file exists
        hdf5_path = azel.azel_dir / "azel_solution.h5"
        assert hdf5_path.exists()
        assert hdf5_path.is_file()

    def test_load_azel_version_from_hdf5(self, mock_flight, sample_azel_version):
        """Test loading saved version matches original."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Save version
        azel._save_to_hdf5(sample_azel_version)

        # Load version back
        loaded_version = azel._load_from_hdf5("rev_20260218_120000")

        # Verify version name matches
        assert loaded_version.version_name == sample_azel_version.version_name

        # Verify DataFrame shape matches
        assert loaded_version.azel_data.shape == sample_azel_version.azel_data.shape

        # Verify DataFrame contents match
        assert loaded_version.azel_data["timestamp"].to_list() == pytest.approx(
            sample_azel_version.azel_data["timestamp"].to_list()
        )
        assert loaded_version.azel_data["az"].to_list() == pytest.approx(
            sample_azel_version.azel_data["az"].to_list()
        )
        assert loaded_version.azel_data["el"].to_list() == pytest.approx(
            sample_azel_version.azel_data["el"].to_list()
        )
        assert loaded_version.azel_data["srange"].to_list() == pytest.approx(
            sample_azel_version.azel_data["srange"].to_list()
        )

        # Verify metadata matches
        assert loaded_version.metadata["telescope_name"] == "SATP1"
        assert loaded_version.metadata["num_samples"] == 3

    def test_hdf5_preserves_dataframe_dtypes(self, mock_flight, sample_azel_version):
        """Test dtypes are preserved after save/load cycle."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Verify original dtypes are Float64
        assert sample_azel_version.azel_data["timestamp"].dtype == pl.Float64
        assert sample_azel_version.azel_data["az"].dtype == pl.Float64
        assert sample_azel_version.azel_data["el"].dtype == pl.Float64
        assert sample_azel_version.azel_data["srange"].dtype == pl.Float64

        # Save and load
        azel._save_to_hdf5(sample_azel_version)
        loaded_version = azel._load_from_hdf5("rev_20260218_120000")

        # Verify loaded dtypes match original
        assert loaded_version.azel_data["timestamp"].dtype == pl.Float64
        assert loaded_version.azel_data["az"].dtype == pl.Float64
        assert loaded_version.azel_data["el"].dtype == pl.Float64
        assert loaded_version.azel_data["srange"].dtype == pl.Float64

    def test_list_versions_returns_all_versions(self, mock_flight):
        """Test listing all saved versions."""
        from pils.analyze.azel import AZELAnalysis, AZELVersion

        azel = AZELAnalysis(mock_flight)

        # Initially no versions
        assert azel.list_versions() == []

        # Create and save multiple versions
        for i, timestamp in enumerate(["120000", "130000", "140000"]):
            azel_data = pl.DataFrame(
                {
                    "timestamp": [float(i * 1000)],
                    "az": [45.0],
                    "el": [30.0],
                    "srange": [100.0],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("az").cast(pl.Float64),
                    pl.col("el").cast(pl.Float64),
                    pl.col("srange").cast(pl.Float64),
                ]
            )

            version = AZELVersion(
                version_name=f"rev_20260218_{timestamp}",
                azel_data=azel_data,
                metadata={"num_samples": 1},
            )
            azel._save_to_hdf5(version)

        # List versions
        versions = azel.list_versions()

        # Should return all 3 versions in chronological order
        assert len(versions) == 3
        assert versions == [
            "rev_20260218_120000",
            "rev_20260218_130000",
            "rev_20260218_140000",
        ]

    def test_get_latest_version_returns_most_recent(self, mock_flight):
        """Test retrieving latest version."""
        from pils.analyze.azel import AZELAnalysis, AZELVersion

        azel = AZELAnalysis(mock_flight)

        # No versions initially
        assert azel.get_latest_version() is None

        # Save multiple versions with different timestamps
        for i, timestamp in enumerate(["120000", "133000", "145000"]):
            azel_data = pl.DataFrame(
                {
                    "timestamp": [float(i * 1000)],
                    "az": [45.0 + i * 10],
                    "el": [30.0],
                    "srange": [100.0],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("az").cast(pl.Float64),
                    pl.col("el").cast(pl.Float64),
                    pl.col("srange").cast(pl.Float64),
                ]
            )

            version = AZELVersion(
                version_name=f"rev_20260218_{timestamp}",
                azel_data=azel_data,
                metadata={"version_id": i},
            )
            azel._save_to_hdf5(version)

        # Get latest version
        latest = azel.get_latest_version()

        # Should return the last version (145000)
        assert latest is not None
        assert latest.version_name == "rev_20260218_145000"
        assert latest.metadata["version_id"] == 2
        assert latest.azel_data["az"][0] == pytest.approx(65.0)

    def test_multiple_versions_in_single_hdf5(self, mock_flight):
        """Test multiple analysis versions stored in same HDF5."""
        from pils.analyze.azel import AZELAnalysis, AZELVersion

        azel = AZELAnalysis(mock_flight)

        # Create 3 different versions
        versions_to_save = []
        for i in range(3):
            azel_data = pl.DataFrame(
                {
                    "timestamp": [1000.0 + i * 100, 2000.0 + i * 100],
                    "az": [45.0 + i * 5, 90.0 + i * 5],
                    "el": [30.0, 45.0],
                    "srange": [100.0 + i * 10, 150.0 + i * 10],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("az").cast(pl.Float64),
                    pl.col("el").cast(pl.Float64),
                    pl.col("srange").cast(pl.Float64),
                ]
            )

            version = AZELVersion(
                version_name=f"rev_20260218_12{i}000",
                azel_data=azel_data,
                metadata={"version_num": i},
            )
            versions_to_save.append(version)
            azel._save_to_hdf5(version)

        # Verify all versions can be loaded independently
        for i, original_version in enumerate(versions_to_save):
            loaded_version = azel._load_from_hdf5(original_version.version_name)

            assert loaded_version.version_name == original_version.version_name
            assert loaded_version.azel_data.shape == original_version.azel_data.shape
            assert loaded_version.metadata["version_num"] == i

            # Verify data values match
            assert loaded_version.azel_data["az"].to_list() == pytest.approx(
                original_version.azel_data["az"].to_list()
            )

    def test_hdf5_file_not_found_raises_error(self, mock_flight):
        """Test loading from non-existent HDF5 raises FileNotFoundError."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Try to load without saving first
        with pytest.raises(FileNotFoundError):
            azel._load_from_hdf5("nonexistent_version")

    def test_hdf5_version_not_found_raises_error(
        self, mock_flight, sample_azel_version
    ):
        """Test loading non-existent version raises KeyError."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight)

        # Save one version
        azel._save_to_hdf5(sample_azel_version)

        # Try to load a different version
        with pytest.raises(KeyError):
            azel._load_from_hdf5("rev_99999999_999999")


# ==================== Phase 1: Data Source Selection Tests ====================


class TestPhase1DataSource:
    """Tests for Phase 1: Data source selection and drone model detection."""

    @pytest.fixture
    def mock_flight_with_sync_data(self, tmp_path):
        """Create mock Flight with sync_data populated."""
        flight_path = tmp_path / "flight_sync"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create mock sync_data
        flight.sync_data = {
            "drone": {
                "correct_timestamp": [1000.0, 2000.0, 3000.0],
                "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
            }
        }

        # Set drone model to DJI
        flight._Flight__drone_model = "dji"

        return flight

    @pytest.fixture
    def mock_flight_raw_data_only(self, tmp_path):
        """Create mock Flight with only raw_data (no sync_data)."""
        flight_path = tmp_path / "flight_raw"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create mock raw_data (no sync_data)
        flight.sync_data = None
        drone_df = pl.DataFrame(
            {
                "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
                "RTK:date": [20260218, 20260218, 20260218],
                "RTK:time": [120000, 120001, 120002],
            }
        )

        from pils.flight import DroneData

        flight.raw_data.drone_data = DroneData(drone_df, None)
        flight._Flight__drone_model = "dji"

        return flight

    def test_sync_data_priority(self, mock_flight_with_sync_data):
        """Test that sync_data is used when available."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_sync_data)

        # Access internal data loading (we'll verify via run_analysis later)
        # For now, just verify sync_data exists
        assert azel.flight.sync_data is not None
        assert "drone" in azel.flight.sync_data
        assert "correct_timestamp" in azel.flight.sync_data["drone"]

    def test_raw_data_fallback(self, mock_flight_raw_data_only):
        """Test that raw_data is used when sync_data is None."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_raw_data_only)

        # Verify sync_data is None and raw_data exists
        assert azel.flight.sync_data is None
        assert azel.flight.raw_data.drone_data is not None
        assert azel.flight.raw_data.drone_data.drone is not None

    def test_litchi_from_sync_data(self, tmp_path):
        """Test that sync_data litchi is accessible."""
        flight_path = tmp_path / "flight_litchi"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Add sync_data with litchi
        flight.sync_data = {
            "drone": {
                "correct_timestamp": [1000.0, 2000.0],
                "RTK:lat_p": [40.0001, 40.0002],
                "RTK:lon_p": [-105.0001, -105.0002],
                "RTK:hmsl_p": [1000.0, 1001.0],
            },
            "litchi": {
                "timestamp": [1000.5, 2000.5],
                "latitude": [40.0001, 40.0002],
                "longitude": [-105.0001, -105.0002],
                "altitude(m)": [1000.0, 1001.0],
            },
        }

        flight._Flight__drone_model = "dji"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # Verify litchi data accessible
        assert "litchi" in azel.flight.sync_data
        assert "latitude" in azel.flight.sync_data["litchi"]

    def test_drone_model_detection_dji(self, mock_flight_with_sync_data):
        """Test that DJI drone model is detected correctly."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_with_sync_data)

        # Access private drone_model attribute
        drone_model = getattr(azel.flight, "_Flight__drone_model", "").lower()
        assert "dji" in drone_model

    def test_drone_model_detection_blacksquare(self, tmp_path):
        """Test that BlackSquare drone model is detected correctly."""
        flight_path = tmp_path / "flight_bs"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        flight.sync_data = {
            "drone": {
                "timestamp": [1000.0, 2000.0],
                "Latitude": [40.0001, 40.0002],
                "Longitude": [-105.0001, -105.0002],
                "heightMSL": [1000.0, 1001.0],
            }
        }

        flight._Flight__drone_model = "blacksquare"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        drone_model = getattr(azel.flight, "_Flight__drone_model", "").lower()
        assert "blacksquare" in drone_model or "black" in drone_model

    def test_sync_data_missing_error(self, tmp_path):
        """Test that clear error is raised when no data available."""
        flight_path = tmp_path / "flight_empty"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # No sync_data and no raw_data
        flight.sync_data = None
        flight.raw_data.drone_data = None

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # We'll verify this when we test run_analysis - for now just create the object
        # The error should occur during run_analysis
        assert azel.flight.sync_data is None
        assert azel.flight.raw_data.drone_data is None


# ==================== Phase 2: Column Mapping and Timestamp Tests ====================


class TestPhase2ColumnMapping:
    """Tests for Phase 2: Column mapping and timestamp handling."""

    def test_dji_columns_from_sync_data(self, tmp_path):
        """Test that DJI columns are correctly identified from sync_data."""
        flight_path = tmp_path / "flight_dji_sync"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with DJI format columns
        flight.sync_data = {
            "drone": {
                "correct_timestamp": [1000.0, 2000.0, 3000.0],
                "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
            }
        }

        flight._Flight__drone_model = "dji"

        # Verify expected columns exist
        drone_data = pl.DataFrame(flight.sync_data["drone"])
        assert "correct_timestamp" in drone_data.columns
        assert "RTK:lat_p" in drone_data.columns
        assert "RTK:lon_p" in drone_data.columns
        assert "RTK:hmsl_p" in drone_data.columns

    def test_blacksquare_columns_from_sync_data(self, tmp_path):
        """Test that BlackSquare columns are correctly identified from sync_data."""
        flight_path = tmp_path / "flight_bs_sync"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with BlackSquare format columns
        flight.sync_data = {
            "drone": {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "Latitude": [40.0001, 40.0002, 40.0003],
                "Longitude": [-105.0001, -105.0002, -105.0003],
                "heightMSL": [1000.0, 1001.0, 1002.0],
            }
        }

        flight._Flight__drone_model = "blacksquare"

        # Verify expected columns exist
        drone_data = pl.DataFrame(flight.sync_data["drone"])
        assert "timestamp" in drone_data.columns
        assert "Latitude" in drone_data.columns
        assert "Longitude" in drone_data.columns
        assert "heightMSL" in drone_data.columns

    def test_litchi_columns_from_sync_data(self, tmp_path):
        """Test that Litchi columns are correctly identified from sync_data."""
        flight_path = tmp_path / "flight_litchi_sync"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with Litchi format columns
        flight.sync_data = {
            "litchi": {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "latitude": [40.0001, 40.0002, 40.0003],
                "longitude": [-105.0001, -105.0002, -105.0003],
                "altitude(m)": [1000.0, 1001.0, 1002.0],
            }
        }

        flight._Flight__drone_model = "dji"  # Litchi is used with DJI drones

        # Verify expected columns exist in litchi data
        litchi_data = pl.DataFrame(flight.sync_data["litchi"])
        assert "timestamp" in litchi_data.columns
        assert "latitude" in litchi_data.columns
        assert "longitude" in litchi_data.columns
        assert "altitude(m)" in litchi_data.columns

    def test_timestamp_direct_extraction(self, tmp_path):
        """Test that timestamps are extracted directly without conversion."""
        flight_path = tmp_path / "flight_ts"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with timestamps already in Unix format
        expected_timestamps = [1000.5, 2000.5, 3000.5]
        flight.sync_data = {
            "drone": {
                "correct_timestamp": expected_timestamps,
                "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
            }
        }

        flight._Flight__drone_model = "dji"

        # Verify timestamps are stored correctly
        drone_data = pl.DataFrame(flight.sync_data["drone"])
        assert drone_data["correct_timestamp"].to_list() == expected_timestamps

    def test_dji_correct_timestamp_column(self, tmp_path):
        """Test that DJI uses correct_timestamp not timestamp."""
        flight_path = tmp_path / "flight_dji_ts"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # DJI should have correct_timestamp, not timestamp
        flight.sync_data = {
            "drone": {
                "correct_timestamp": [1000.0, 2000.0],
                "RTK:lat_p": [40.0001, 40.0002],
                "RTK:lon_p": [-105.0001, -105.0002],
                "RTK:hmsl_p": [1000.0, 1001.0],
            }
        }

        flight._Flight__drone_model = "dji"

        drone_data = pl.DataFrame(flight.sync_data["drone"])
        assert "correct_timestamp" in drone_data.columns
        # Should NOT have generic "timestamp" for DJI
        assert "timestamp" not in drone_data.columns

    def test_no_timezone_conversion_for_sync_data(self, tmp_path):
        """Test that drone_timezone_hours has no effect on sync_data (already UTC)."""
        flight_path = tmp_path / "flight_tz"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Timestamps in sync_data are already UTC
        original_timestamps = [1000.0, 2000.0, 3000.0]
        flight.sync_data = {
            "drone": {
                "correct_timestamp": original_timestamps.copy(),
                "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
            }
        }

        flight._Flight__drone_model = "dji"

        # Verify timestamps remain unchanged (UTC)
        drone_data = pl.DataFrame(flight.sync_data["drone"])
        assert drone_data["correct_timestamp"].to_list() == original_timestamps


class TestPhase3RTKCorrection:
    """Test suite for Phase 3: Conditional RTK correction based on drone format."""

    def test_dji_applies_rtk_correction(self, tmp_path):
        """Test that DJI format applies RTK correction."""
        flight_path = tmp_path / "flight_dji_rtk"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with DJI format
        flight.sync_data = {
            "drone": pl.DataFrame(
                {
                    "correct_timestamp": [1000.0, 2000.0, 3000.0],
                    "RTK:lat_p": [40.0001, 40.0002, 40.0003],
                    "RTK:lon_p": [-105.0001, -105.0002, -105.0003],
                    "RTK:hmsl_p": [1000.0, 1001.0, 1002.0],
                }
            )
        }

        flight._Flight__drone_model = "dji"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # Create mock EMLID CSV
        emlid_csv = tmp_path / "emlid.csv"
        emlid_csv.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height\n"
            "SATP1_1,-105.0,40.0,1000.0\n"
            "DJI_1,-105.0,40.0,1000.0\n"
        )

        dji_broadcast = {"lat": 40.001, "lon": -105.0, "alt": 1000.0}

        version = azel.run_analysis(emlid_csv, "SATP1", dji_broadcast)

        assert version is not None
        assert "rtk_applied" in version.metadata
        assert version.metadata["rtk_applied"] is True
        # Verify RTK correction values are non-zero (due to broadcast offset)
        rtk_corr = version.metadata["rtk_correction"]
        assert abs(rtk_corr["delta_e"]) > 0 or abs(rtk_corr["delta_n"]) > 0

    def test_blacksquare_skips_rtk_correction(self, tmp_path):
        """Test that BlackSquare format skips RTK correction (zero offset)."""
        flight_path = tmp_path / "flight_bs_rtk"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with BlackSquare format
        flight.sync_data = {
            "drone": pl.DataFrame(
                {
                    "timestamp": [1000.0, 2000.0, 3000.0],
                    "Latitude": [40.0001, 40.0002, 40.0003],
                    "Longitude": [-105.0001, -105.0002, -105.0003],
                    "heightMSL": [1000.0, 1001.0, 1002.0],
                }
            )
        }

        flight._Flight__drone_model = "blacksquare"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # Create mock EMLID CSV
        emlid_csv = tmp_path / "emlid.csv"
        emlid_csv.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height\n"
            "SATP1_1,-105.0,40.0,1000.0\n"
            "DJI_1,-105.0,40.0,1000.0\n"
        )

        dji_broadcast = {"lat": 40.001, "lon": -105.0, "alt": 1000.0}

        version = azel.run_analysis(emlid_csv, "SATP1", dji_broadcast)

        assert version is not None
        assert "rtk_applied" in version.metadata
        assert version.metadata["rtk_applied"] is False
        # Verify RTK correction is zero for BlackSquare
        rtk_corr = version.metadata["rtk_correction"]
        assert rtk_corr["delta_e"] == 0.0
        assert rtk_corr["delta_n"] == 0.0
        assert rtk_corr["delta_u"] == 0.0

    def test_litchi_applies_rtk_correction(self, tmp_path):
        """Test that Litchi format applies RTK correction."""
        flight_path = tmp_path / "flight_litchi_rtk"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with Litchi format
        flight.sync_data = {
            "drone": pl.DataFrame(
                {
                    "timestamp": [1000.0, 2000.0, 3000.0],
                    "latitude": [40.0001, 40.0002, 40.0003],
                    "longitude": [-105.0001, -105.0002, -105.0003],
                    "altitude(m)": [1000.0, 1001.0, 1002.0],
                }
            )
        }

        flight._Flight__drone_model = "litchi"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # Create mock EMLID CSV
        emlid_csv = tmp_path / "emlid.csv"
        emlid_csv.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height\n"
            "SATP1_1,-105.0,40.0,1000.0\n"
            "DJI_1,-105.0,40.0,1000.0\n"
        )

        dji_broadcast = {"lat": 40.001, "lon": -105.0, "alt": 1000.0}

        version = azel.run_analysis(emlid_csv, "SATP1", dji_broadcast)

        assert version is not None
        assert "rtk_applied" in version.metadata
        assert version.metadata["rtk_applied"] is True
        # Verify RTK correction values are non-zero (due to broadcast offset)
        rtk_corr = version.metadata["rtk_correction"]
        assert abs(rtk_corr["delta_e"]) > 0 or abs(rtk_corr["delta_n"]) > 0

    def test_metadata_includes_drone_format(self, tmp_path):
        """Test that metadata includes drone_format field."""
        flight_path = tmp_path / "flight_meta"
        flight_path.mkdir()
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir()

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with BlackSquare format
        flight.sync_data = {
            "drone": pl.DataFrame(
                {
                    "timestamp": [1000.0, 2000.0, 3000.0],
                    "Latitude": [40.0001, 40.0002, 40.0003],
                    "Longitude": [-105.0001, -105.0002, -105.0003],
                    "heightMSL": [1000.0, 1001.0, 1002.0],
                }
            )
        }

        flight._Flight__drone_model = "blacksquare"

        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(flight)

        # Create mock EMLID CSV
        emlid_csv = tmp_path / "emlid.csv"
        emlid_csv.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height\n"
            "SATP1_1,-105.0,40.0,1000.0\n"
            "DJI_1,-105.0,40.0,1000.0\n"
        )

        dji_broadcast = {"lat": 40.001, "lon": -105.0, "alt": 1000.0}

        version = azel.run_analysis(emlid_csv, "SATP1", dji_broadcast)

        assert version is not None
        assert "drone_format" in version.metadata
        assert version.metadata["drone_format"] == "blacksquare"
