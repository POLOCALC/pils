"""Tests for EMLID reference coordinate loader."""

import polars as pl
import pytest


class TestEmlidLoader:
    """Test EMLID coordinate loading and barycenter computation."""

    @pytest.fixture
    def mock_emlid_campaign(self, tmp_path):
        """Create campaign structure with EMLID CSV for testing.

        Creates directory structure:
        campaign/
            flight_001/
                drone/
            metadata/
                202511_coordinates.csv
        """
        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()

        # Create flight directory
        flight_dir = campaign_dir / "flight_001"
        drone_dir = flight_dir / "drone"
        drone_dir.mkdir(parents=True)

        # Create metadata directory with EMLID CSV
        metadata_dir = campaign_dir / "metadata"
        metadata_dir.mkdir()

        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.0500,40.0500,1050.0
SATP1_01,-105.0000,40.0000,1000.0
SATP1_02,-105.0001,40.0001,1000.5
SATP1_03,-105.0002,40.0002,1001.0
dji rtk base (antenna base),-105.1000,40.1000,1100.0
DJI RTK BASE (antenna base) 2,-105.1001,40.1001,1100.5
"""
        csv_file = metadata_dir / "202511_coordinates.csv"
        csv_file.write_text(csv_content)

        return flight_dir, csv_file

    def test_emlid_initialization_finds_csv(self, mock_emlid_campaign):
        """Test Emlid finds CSV file at expected location."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, csv_file = mock_emlid_campaign

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        assert emlid.emlid_path == csv_file
        assert emlid.emlid_path.exists()

    def test_emlid_initialization_raises_if_csv_missing(self, tmp_path):
        """Test that missing EMLID file raises FileNotFoundError."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        # Create flight structure WITHOUT metadata directory
        campaign_dir = tmp_path / "campaign"
        flight_dir = campaign_dir / "flight_001"
        drone_dir = flight_dir / "drone"
        drone_dir.mkdir(parents=True)

        flight = Flight({"drone_data_folder_path": str(drone_dir)})

        with pytest.raises(FileNotFoundError, match="EMLID CSV file not found"):
            Emlid(flight)

    def test_load_data_returns_telescope_position(self, mock_emlid_campaign):
        """Test loading telescope position from EMLID CSV."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, _ = mock_emlid_campaign

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        ref_data = emlid.load_data(telescope_name="SATP1")

        # Check telescope key exists
        assert "telescope" in ref_data
        telescope = ref_data["telescope"]
        assert isinstance(telescope, pl.DataFrame)
        assert telescope.height == 1  # Single row DataFrame

        # Extract values
        telescope_dict = telescope.row(0, named=True)
        assert "lat" in telescope_dict
        assert "lon" in telescope_dict
        assert "alt" in telescope_dict

        # Check barycenter values (computed in ENU space)
        # Should be close to simple mean but not exactly the same
        assert telescope_dict["lat"] == pytest.approx(40.0001, abs=0.001)
        assert telescope_dict["lon"] == pytest.approx(-105.0001, abs=0.001)
        assert telescope_dict["alt"] == pytest.approx(1000.5, abs=1.0)

    def test_load_data_returns_base_positions(self, mock_emlid_campaign):
        """Test loading EMLID and DJI base positions from EMLID CSV."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, _ = mock_emlid_campaign

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        ref_data = emlid.load_data(telescope_name="SATP1")

        # Check base key exists with nested dict
        assert "base" in ref_data
        assert isinstance(ref_data["base"], dict)
        assert "emlid" in ref_data["base"]
        assert "dji" in ref_data["base"]

        # Check EMLID base (single measurement, should be exact)
        emlid_base = ref_data["base"]["emlid"]
        assert isinstance(emlid_base, pl.DataFrame)
        assert emlid_base.height == 1
        emlid_dict = emlid_base.row(0, named=True)
        assert emlid_dict["lat"] == pytest.approx(40.0500, abs=0.0001)
        assert emlid_dict["lon"] == pytest.approx(-105.0500, abs=0.0001)
        assert emlid_dict["alt"] == pytest.approx(1050.0, abs=0.01)

        # Check DJI base (barycenter of 2 measurements)
        dji_base = ref_data["base"]["dji"]
        assert isinstance(dji_base, pl.DataFrame)
        assert dji_base.height == 1
        dji_dict = dji_base.row(0, named=True)
        assert dji_dict["lat"] == pytest.approx(40.10005, abs=0.001)
        assert dji_dict["lon"] == pytest.approx(-105.10005, abs=0.001)
        assert dji_dict["alt"] == pytest.approx(1100.25, abs=1.0)

    def test_load_data_filters_by_telescope_name(self, tmp_path):
        """Test filtering by telescope name prefix."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        # Create campaign structure
        campaign_dir = tmp_path / "campaign"
        flight_dir = campaign_dir / "flight_001"
        drone_dir = flight_dir / "drone"
        drone_dir.mkdir(parents=True)

        metadata_dir = campaign_dir / "metadata"
        metadata_dir.mkdir()

        # Create CSV with multiple telescope types
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.2500,40.2500,1250.0
SATP1_01,-105.0000,40.0000,1000.0
SATP1_02,-105.0001,40.0001,1001.0
SATP2_01,-106.0000,41.0000,2000.0
SATP2_02,-106.0001,41.0001,2001.0
dji rtk base (antenna base),-105.5000,40.5000,1500.0
"""
        csv_file = metadata_dir / "202511_coordinates.csv"
        csv_file.write_text(csv_content)

        flight = Flight({"drone_data_folder_path": str(drone_dir)})
        emlid = Emlid(flight)

        # Load SATP1 data
        ref_data_1 = emlid.load_data(telescope_name="SATP1")
        telescope_1 = ref_data_1["telescope"].row(0, named=True)
        assert telescope_1["lat"] == pytest.approx(40.00005, abs=0.001)
        assert telescope_1["lon"] == pytest.approx(-105.00005, abs=0.001)

        # Load SATP2 data
        ref_data_2 = emlid.load_data(telescope_name="SATP2")
        telescope_2 = ref_data_2["telescope"].row(0, named=True)
        assert telescope_2["lat"] == pytest.approx(41.00005, abs=0.001)
        assert telescope_2["lon"] == pytest.approx(-106.00005, abs=0.001)

        # EMLID base should be the same for both
        emlid_base_1 = ref_data_1["base"]["emlid"].row(0, named=True)
        emlid_base_2 = ref_data_2["base"]["emlid"].row(0, named=True)
        assert emlid_base_1["lat"] == emlid_base_2["lat"]

    def test_load_data_raises_if_telescope_not_found(self, mock_emlid_campaign):
        """Test that missing telescope name raises ValueError."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, csv_file = mock_emlid_campaign

        # Overwrite CSV with only base data (no telescope)
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.0000,40.0000,1000.0
dji rtk base (antenna base),-105.1000,40.1000,1100.0
"""
        csv_file.write_text(csv_content)

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        with pytest.raises(
            ValueError, match="No telescope positions found for 'SATP1'"
        ):
            emlid.load_data(telescope_name="SATP1")

    def test_load_data_raises_if_dji_base_not_found(self, mock_emlid_campaign):
        """Test that missing DJI base positions raises ValueError."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, csv_file = mock_emlid_campaign

        # Overwrite CSV with only telescope data (no DJI base)
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.0000,40.0000,1000.0
SATP1_01,-105.0000,40.0000,1000.0
"""
        csv_file.write_text(csv_content)

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        with pytest.raises(ValueError, match="No DJI base positions found"):
            emlid.load_data(telescope_name="SATP1")

    def test_barycenter_computation_with_multiple_measurements(
        self, mock_emlid_campaign
    ):
        """Test that barycenter is computed for multiple telescope measurements."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, _ = mock_emlid_campaign

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        # SATP1 has 3 measurements, so barycenter should be computed
        ref_data = emlid.load_data(telescope_name="SATP1")
        telescope = ref_data["telescope"]

        # Should return single row (barycenter)
        assert telescope.height == 1

        # Barycenter should be approximately at the mean
        # (not exactly mean due to ENU transformation)
        telescope_dict = telescope.row(0, named=True)

        # Approximate mean lat: (40.0000 + 40.0001 + 40.0002) / 3 ≈ 40.0001
        assert telescope_dict["lat"] == pytest.approx(40.0001, abs=0.001)
        # Approximate mean lon: (-105.0000 + -105.0001 + -105.0002) / 3 ≈ -105.0001
        assert telescope_dict["lon"] == pytest.approx(-105.0001, abs=0.001)
        # Approximate mean alt: (1000.0 + 1000.5 + 1001.0) / 3 ≈ 1000.5
        assert telescope_dict["alt"] == pytest.approx(1000.5, abs=1.0)

    def test_single_measurement_returns_exact_value(self, mock_emlid_campaign):
        """Test that single measurement returns exact value (no barycenter needed)."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        flight_dir, csv_file = mock_emlid_campaign

        # Overwrite CSV with single measurement for telescope
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.0500,40.0500,1050.0
SATP1_01,-105.1234,40.5678,1234.56
dji rtk base (antenna base),-105.1000,40.1000,1100.0
"""
        csv_file.write_text(csv_content)

        flight = Flight({"drone_data_folder_path": str(flight_dir / "drone")})
        emlid = Emlid(flight)

        ref_data = emlid.load_data(telescope_name="SATP1")
        telescope = ref_data["telescope"]

        # Should return single row with exact values
        assert telescope.height == 1
        telescope_dict = telescope.row(0, named=True)

        # Should match exactly (no barycenter computation)
        assert telescope_dict["lat"] == pytest.approx(40.5678, abs=0.0001)
        assert telescope_dict["lon"] == pytest.approx(-105.1234, abs=0.0001)
        assert telescope_dict["alt"] == pytest.approx(1234.56, abs=0.01)

    def test_custom_base_names(self, tmp_path):
        """Test loading with custom base station names."""
        from pils.flight import Flight
        from pils.sensors.emlid import Emlid

        # Create campaign structure
        campaign_dir = tmp_path / "campaign"
        flight_dir = campaign_dir / "flight_001"
        drone_dir = flight_dir / "drone"
        drone_dir.mkdir(parents=True)

        metadata_dir = campaign_dir / "metadata"
        metadata_dir.mkdir()

        # Create CSV with custom base names
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
my custom base,-105.0500,40.0500,1050.0
SATP1_01,-105.0000,40.0000,1000.0
my dji base,-105.1000,40.1000,1100.0
"""
        csv_file = metadata_dir / "202511_coordinates.csv"
        csv_file.write_text(csv_content)

        flight = Flight({"drone_data_folder_path": str(drone_dir)})
        emlid = Emlid(flight)

        # Load with custom base names
        ref_data = emlid.load_data(
            telescope_name="SATP1",
            base_name="my custom base",
            dji_base_name="my dji base",
        )

        # Should find both bases
        assert "base" in ref_data
        assert "emlid" in ref_data["base"]
        assert "dji" in ref_data["base"]

        emlid_base = ref_data["base"]["emlid"].row(0, named=True)
        assert emlid_base["lat"] == pytest.approx(40.0500, abs=0.0001)

        dji_base = ref_data["base"]["dji"].row(0, named=True)
        assert dji_base["lat"] == pytest.approx(40.1000, abs=0.0001)
