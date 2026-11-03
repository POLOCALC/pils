"""pils.integrations.stout: the one module that imports STOUT."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pils.integrations.stout as stout_db

# Imported here, not under a patched sys.modules (which drops what a test
# imports when it is restored)
from pils.flight import Flight
from pils.lab_analysis.store import DEFAULT_LAB_ROOT, default_lab_root

STOUT_MODULES = (
    "stout",
    "stout.config",
    "stout.exceptions",
    "stout.services",
    "stout.services.campaigns",
    "stout.services.inventory",
    "stout.services.inventory.service",
)


@pytest.fixture(autouse=True)
def fresh_services():
    stout_db._services.clear()
    yield
    stout_db._services.clear()


@pytest.fixture
def no_stout():
    with patch.dict(sys.modules, dict.fromkeys(STOUT_MODULES)):
        yield


class NotFoundError(Exception):
    pass


@pytest.fixture
def fake_stout(tmp_path):
    """A stand-in STOUT: yields (campaign service, inventory items)."""
    campaigns = MagicMock()
    items = {
        "drone-1": {"name": "DRN_001", "specifications": {"model": "Matrice 300"}},
        "drone-2": {"name": "BlackSquare X", "specifications": {}},
    }

    class InventoryService:
        def get_item_by_id(self, item_id):
            if item_id not in items:
                raise NotFoundError(item_id)
            return items[item_id]

    modules = {
        "stout": types.ModuleType("stout"),
        "stout.config": types.SimpleNamespace(
            Config=types.SimpleNamespace(data_root=lambda: tmp_path)
        ),
        "stout.exceptions": types.SimpleNamespace(NotFoundError=NotFoundError),
        "stout.services": types.ModuleType("stout.services"),
        "stout.services.campaigns": types.SimpleNamespace(
            CampaignService=lambda: campaigns
        ),
        "stout.services.inventory": types.ModuleType("stout.services.inventory"),
        "stout.services.inventory.service": types.SimpleNamespace(
            InventoryService=InventoryService
        ),
    }
    with patch.dict(sys.modules, modules):
        yield campaigns, items


class TestWithoutStout:
    def test_not_available(self, no_stout):
        assert stout_db.is_available() is False

    @pytest.mark.parametrize(
        "call",
        [
            lambda: stout_db.data_path(),
            lambda: stout_db.all_campaigns(),
            lambda: stout_db.all_flights(),
            lambda: stout_db.campaign_flights("c"),
            lambda: stout_db.flight(flight_id="f"),
            lambda: stout_db.payload("p"),
            lambda: stout_db.inventory_item("i"),
            lambda: stout_db.drone_model("d"),
        ],
    )
    def test_every_query_raises_stout_unavailable(self, no_stout, call):
        with pytest.raises(stout_db.StoutUnavailable, match=r"pils\[stout\]"):
            call()

    def test_stout_unavailable_is_an_import_error(self):
        assert issubclass(stout_db.StoutUnavailable, ImportError)

    def test_lab_root_falls_back(self, no_stout, monkeypatch):
        monkeypatch.delenv("PILS_LAB_DATA_PATH", raising=False)
        assert default_lab_root() == DEFAULT_LAB_ROOT

    def test_flight_drone_model_falls_back_to_the_folder(self, no_stout, tmp_path):
        flight = Flight({"drone_data_folder_path": str(tmp_path), "drone_id": "d"})
        assert flight._detect_drone_model(str(tmp_path)) == "dji"


class TestWithAStandIn:
    def test_available_and_data_path(self, fake_stout, tmp_path):
        assert stout_db.is_available() is True
        assert stout_db.data_path() == tmp_path

    def test_campaign_queries(self, fake_stout):
        campaigns, _ = fake_stout
        campaigns.get_all_campaigns.return_value = [{"name": "202511"}]
        campaigns.get_flights_by_campaign.return_value = [{"flight_name": "f"}]
        campaigns.get_payload_by_id.return_value = {"payload_id": "p"}
        assert stout_db.all_campaigns() == [{"name": "202511"}]
        assert stout_db.campaign_flights("c1") == [{"flight_name": "f"}]
        campaigns.get_flights_by_campaign.assert_called_once_with(campaign_id="c1")
        assert stout_db.payload("p") == {"payload_id": "p"}
        stout_db.flight(flight_name="n")
        campaigns.get_flight.assert_called_once_with(flight_name="n", flight_id=None)

    def test_inventory(self, fake_stout):
        assert stout_db.inventory_item("drone-1")["name"] == "DRN_001"
        assert stout_db.drone_model("drone-1") == "matrice 300"
        assert stout_db.drone_model("drone-2") == "blacksquare x"
        with pytest.raises(stout_db.StoutNotFound):
            stout_db.inventory_item("missing")

    def test_flight_drone_model_from_the_inventory(self, fake_stout, tmp_path):
        for drone_id, model in (("drone-1", "dji"), ("drone-2", "blacksquare")):
            flight = Flight(
                {"drone_data_folder_path": str(tmp_path), "drone_id": drone_id}
            )
            assert flight._detect_drone_model(str(tmp_path)) == model


def test_only_the_adapter_imports_stout():
    root = Path(stout_db.__file__).parents[1]
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path != Path(stout_db.__file__)
        and any(
            line.strip().startswith(("import stout", "from stout"))
            for line in path.read_text().splitlines()
        )
    ]
    assert offenders == []


def _stout_package() -> Path | None:
    spec = importlib.util.find_spec("stout")
    if spec is None or spec.origin is None:  # not installed, or a bare folder
        return None
    return Path(spec.origin).parent


@pytest.mark.skipif(_stout_package() is None, reason="STOUT is not installed")
def test_with_stout_on_a_throwaway_database(tmp_path):
    """The real STOUT, on an SQLite database in tmp_path (in a subprocess, so
    the STOUT_* variables are read before STOUT is imported and the live
    database is never opened)."""
    package = _stout_package()
    assert package is not None
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    for schema in (package / "schemas").glob("*.json"):
        (schemas / schema.name).write_text(schema.read_text())
    (tmp_path / "data").mkdir()
    env = {
        **os.environ,
        "STOUT_DB_PATH": str(tmp_path / "db.sqlite"),
        "STOUT_DATA_PATH": str(tmp_path / "data"),
        "STOUT_LOG_DIR": str(tmp_path / "logs"),
        "STOUT_SCHEMAS_DIR": str(schemas),
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
    }
    script = textwrap.dedent(
        """
        import json
        import pils.integrations.stout as stout_db
        from stout.services.campaigns import CampaignService
        from stout.services.inventory.service import InventoryService

        item = InventoryService().create_item(
            "drone", specifications={"model": "Matrice 300 RTK", "manufacturer": "DJI"}
        )
        campaign = CampaignService().create_campaign("202511")
        try:
            stout_db.inventory_item("missing")
            missing = False
        except stout_db.StoutNotFound:
            missing = True
        print(json.dumps({
            "available": stout_db.is_available(),
            "data_path": str(stout_db.data_path()),
            "model": stout_db.drone_model(item["item_id"]),
            "campaigns": [c["campaign_id"] for c in stout_db.all_campaigns()],
            "campaign": campaign["campaign_id"],
            "flights": stout_db.campaign_flights(campaign["campaign_id"]),
            "flight": stout_db.flight(flight_id="none"),
            "missing": missing,
        }))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["available"] is True
    assert result["data_path"] == str(tmp_path / "data")
    assert result["model"] == "matrice 300 rtk"
    assert result["campaigns"] == [result["campaign"]]
    assert result["flights"] == []
    assert result["flight"] is None
    assert result["missing"] is True
