import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import requests
import urllib3


class CloudnetDataset:
    def __init__(self, root_dir: str, site: str, date_from: str, date_to: str):
        self.session = Session()
        self.root_dir = root_dir
        rpg_records = self.session.get_rpg_records(site, date_from, date_to)
        classification_records = self.session.get_classification_records(
            site, date_from, date_to
        )
        self.records = _group_records_by_date(rpg_records + classification_records)

    def __iter__(self) -> Iterable[tuple[str, list[str], list[str]]]:
        for date, records in sorted(self.records.items()):
            rpg_files = [
                self.get_file(r, "rpg")
                for r in sorted(records["rpg"], key=lambda r: r["filename"])
            ]
            classification_files = [
                self.get_file(r, "classification")
                for r in sorted(records["classification"], key=lambda r: r["filename"])
            ]
            yield date, rpg_files, classification_files

    def get_file(self, record: dict, record_type: str) -> str:
        if record_type == "rpg":
            pth = self._rpg_record2path(record)
        elif record_type == "classification":
            pth = self._classification_record2path(record)
        else:
            raise ValueError
        if not pth.exists():
            res = self.session.get(record["downloadUrl"])
            fname = record["filename"]
            logging.info(f"Downloaded {fname}")
            pth.parent.mkdir(parents=True, exist_ok=True)
            with pth.open("wb") as f:
                f.write(res.content)
        return str(pth)

    def _rpg_record2path(self, record: dict) -> Path:
        return Path(
            self.root_dir,
            record["siteId"],
            record["measurementDate"],
            record["instrumentId"],
            record["filename"],
        )

    def _classification_record2path(self, record: dict) -> Path:
        return Path(
            self.root_dir,
            record["site"]["id"],
            record["measurementDate"],
            "classification",
            record["filename"],
        )


def _group_records_by_date(records: list[dict]) -> dict:
    dd: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("instrumentId") == "rpg-fmcw-94":
            dd[r["measurementDate"]]["rpg"].append(r)
        elif r.get("product", {})["id"] == "classification":
            dd[r["measurementDate"]]["classification"].append(r)
        else:
            raise ValueError
    return dict(dd)


class Session(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        retries = urllib3.util.retry.Retry(total=10, backoff_factor=0.2)
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.mount("https://", adapter)

    def get_rpg_records(self, site: str, date_from: str, date_to: str) -> list[dict]:
        url = "https://cloudnet.fmi.fi/api/raw-files"
        params = {
            "site": site,
            "dateFrom": date_from,
            "dateTo": date_to,
            "instrument": "rpg-fmcw-94",
        }
        records = self.get(url, params=params).json()
        return [r for r in records if r["filename"].endswith(".LV0")]

    def get_classification_records(
        self, site: str, date_from: str, date_to: str
    ) -> list[dict]:
        url = "https://cloudnet.fmi.fi/api/files"
        params = {
            "site": site,
            "dateFrom": date_from,
            "dateTo": date_to,
            "product": "classification",
        }
        return self.get(url, params=params).json()
