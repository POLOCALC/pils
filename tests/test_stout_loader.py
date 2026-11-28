"""Tests for StoutLoader."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pils.loader.stout import StoutLoader


def stout_unavailable():
    """Make every ``stout`` import fail, as when STOUT is not installed."""
    return patch.dict(
        sys.modules,
        {
            "stout": None,
            "stout.config": None,
            "stout.services": None,
            "stout.services.campaigns": None,
        },
    )


@pytest.fixture
def stout_available():
    """Install a stand-in ``stout`` package; yields its CampaignService."""
    service = MagicMock()
    config = types.SimpleNamespace(data_root=lambda: Path("/data"))
    modules = {
        "stout": types.ModuleType("stout"),
        "stout.config": types.SimpleNamespace(Config=config),
        "stout.services": types.ModuleType("stout.services"),
        "stout.services.campaigns": types.SimpleNamespace(
            CampaignService=lambda: service
        ),
    }
    with patch.dict(sys.modules, modules):
        yield service


class TestStoutLoaderInitialization:
    """Test StoutLoader initialization."""

    def test_init_attempts_stout_import(self):
        """Test initialization attempts to import STOUT."""
        with stout_unavailable():
            loader = StoutLoader()

            # Should fall back to filesystem mode
            assert loader.use_stout is False
            assert loader.base_data_path is None

    def test_init_with_stout(self, stout_available):
        """Test the loader uses the STOUT database when it is installed."""
        loader = StoutLoader()

        assert loader.use_stout is True
        assert str(loader.base_data_path) == "/data"

    def test_flights_come_from_stout(self, stout_available):
        stout_available.get_flights_by_campaign.return_value = [{"flight_name": "f"}]
        stout_available.get_flight.return_value = {"flight_name": "g"}

        loader = StoutLoader()

        assert loader.load_all_campaign_flights(campaign_id="c1") == [
            {"flight_name": "f"}
        ]
        stout_available.get_flights_by_campaign.assert_called_with(campaign_id="c1")
        assert loader.load_single_flight(flight_id="x") == {"flight_name": "g"}

    def test_database_queries_without_stout_raise(self):
        with stout_unavailable():
            loader = StoutLoader()
            with pytest.raises(RuntimeError):
                loader.load_all_flights()
