from dataclasses import dataclass
from pathlib import Path
import re, datetime as dt
from typing import Optional

@dataclass
class EOImage:
    path: str
    sensor: Optional[str] = None
    satellite: Optional[str] = None
    modality: Optional[str] = None
    resolution: Optional[str] = None
    date: Optional[str] = None
    cloud: Optional[int] = None
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

_date_re = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_sensor_re = re.compile(r"(S1|S2|L8|L9|PL|WV|GeoEye|SPOT)", re.I)
_cloud_re = re.compile(r"cloud(?:_|-)?(\d{1,2})", re.I)

def infer_metadata(img_path: str) -> EOImage:
    name = Path(img_path).name
    date = next(iter(_date_re.findall(name)), None)
    sensor = (m.group(1).upper() if (m:=_sensor_re.search(name)) else None)
    cloud = int(m.group(1)) if (m:=_cloud_re.search(name)) else None
    # basic ISO fallback
    if date:
        try: dt.date.fromisoformat(date)
        except: date = None
    return EOImage(path=img_path, sensor=sensor or "unknown", date=date or "unknown", cloud=cloud or 0)
