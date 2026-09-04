"""
gtfs_departures.py
------------------
Reads Magdeburg GTFS data and returns upcoming departures
for the two university tram stops.

Used by wayfinder_app.py — imported once on startup,
data is cached in memory for fast repeated queries.
"""

import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone


# -------------------------
#  STOP CONFIGURATION
# -------------------------

UNIVERSITY_STOPS = {
    "universitaet": {
        "label": "Magdeburg, Universität",
        "icon":  "🎓",
        "ids":   {"de:15003:19686::01", "de:15003:19686::02"},
    },
    "bibliothek": {
        "label": "Magdeburg, Universitätsbibliothek",
        "icon":  "📚",
        "ids":   {"de:15003:7482::01", "de:15003:7482::02",
                  "de:15003:7482::04", "de:15003:7482::05"},
    },
}

ALL_STOP_IDS = set()
for s in UNIVERSITY_STOPS.values():
    ALL_STOP_IDS.update(s["ids"])


# -------------------------
#  GTFS LOADER
# -------------------------

class GTFSDepartures:
    """
    Loads GTFS zip once, caches stop_times + trips + routes + calendar.
    Call .next_departures() to get upcoming trams for university stops.
    """

    def __init__(self, gtfs_zip_path: str):
        self.gtfs_path = gtfs_zip_path
        self._stop_times = []   # list of dicts
        self._trips      = {}   # trip_id -> {route_id, service_id, headsign}
        self._routes     = {}   # route_id -> {short_name, long_name}
        self._calendar   = {}   # service_id -> set of weekday names
        self._cal_dates  = {}   # service_id -> {date: exception_type}
        self._loaded     = False

    def load(self):
        """Parse GTFS zip into memory. Call once on startup."""
        with zipfile.ZipFile(self.gtfs_path) as z:
            self._load_routes(z)
            self._load_trips(z)
            self._load_calendar(z)
            self._load_calendar_dates(z)
            self._load_stop_times(z)
        self._loaded = True

    def _read(self, z, filename):
        with z.open(filename) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))

    def _load_routes(self, z):
        for row in self._read(z, "routes.txt"):
            self._routes[row["route_id"]] = {
                "short_name": row.get("route_short_name", ""),
                "long_name":  row.get("route_long_name", ""),
            }

    def _load_trips(self, z):
        for row in self._read(z, "trips.txt"):
            self._trips[row["trip_id"]] = {
                "route_id":    row.get("route_id", ""),
                "service_id":  row.get("service_id", ""),
                "headsign":    row.get("trip_headsign", ""),
            }

    def _load_calendar(self, z):
        days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        for row in self._read(z, "calendar.txt"):
            active = {d for d in days if row.get(d) == "1"}
            self._calendar[row["service_id"]] = {
                "days":       active,
                "start_date": row.get("start_date", ""),
                "end_date":   row.get("end_date", ""),
            }

    def _load_calendar_dates(self, z):
        for row in self._read(z, "calendar_dates.txt"):
            sid  = row["service_id"]
            date = row["date"]
            etype = row.get("exception_type", "1")
            if sid not in self._cal_dates:
                self._cal_dates[sid] = {}
            self._cal_dates[sid][date] = etype

    def _load_stop_times(self, z):
        for row in self._read(z, "stop_times.txt"):
            if row.get("stop_id") in ALL_STOP_IDS:
                self._stop_times.append({
                    "trip_id":        row["trip_id"],
                    "stop_id":        row["stop_id"],
                    "departure_time": row.get("departure_time", ""),
                })

    # -------------------------
    #  SERVICE DAY CHECK
    # -------------------------

    def _is_service_active(self, service_id: str, date: datetime) -> bool:
        """
        Check if a service_id runs on the given date.
        We use the base calendar only — calendar_dates exceptions
        (type 2 = cancelled) are ignored so we always show the
        scheduled timetable. A warning is added at the response level
        if all services are cancelled today.
        """
        date_str = date.strftime("%Y%m%d")
        weekday  = date.strftime("%A").lower()

        # Only honour type 1 (added) exceptions — never remove services
        if service_id in self._cal_dates:
            exc = self._cal_dates[service_id].get(date_str)
            if exc == "1":
                return True
            # type 2 (cancelled) — fall through to base calendar anyway

        # Base calendar
        cal = self._calendar.get(service_id)
        if not cal:
            return False

        if date_str < cal["start_date"] or date_str > cal["end_date"]:
            return False

        return weekday in cal["days"]

    # -------------------------
    #  PARSE TIME
    # -------------------------

    @staticmethod
    def _parse_gtfs_time(time_str: str, base_date: datetime) -> datetime | None:
        """
        Parse GTFS time string (HH:MM:SS) into a datetime.
        GTFS allows hours >= 24 for trips past midnight.
        """
        try:
            parts = time_str.strip().split(":")
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return base_date.replace(hour=0, minute=0, second=0, microsecond=0) \
                   + timedelta(hours=h, minutes=m, seconds=s)
        except Exception:
            return None

    # -------------------------
    #  NEXT DEPARTURES
    # -------------------------

    def next_departures(self, limit: int = 6, window_minutes: int = 90) -> dict:
        """
        Return next departures from both university stops.

        Args:
            limit:          Max departures per stop.
            window_minutes: How far ahead to look.

        Returns:
            {
              "generated_at": str,
              "stops": {
                "universitaet": { label, icon, departures: [...] },
                "bibliothek":   { label, icon, departures: [...] },
              }
            }
        """
        if not self._loaded:
            self.load()

        now      = datetime.now()
        cutoff   = now + timedelta(minutes=window_minutes)
        today    = now.date()

        # Build lookup: trip_id -> is_active_today
        active_trips = {}
        for trip_id, trip in self._trips.items():
            sid = trip["service_id"]
            active_trips[trip_id] = self._is_service_active(sid, now)

        # Collect upcoming departures per stop group
        result = {}
        for group_key, group in UNIVERSITY_STOPS.items():
            departures = []

            for st in self._stop_times:
                trip_id = st["trip_id"]
                stop_id = st["stop_id"]

                if stop_id not in group["ids"]:
                    continue
                if not active_trips.get(trip_id, False):
                    continue

                dep_dt = self._parse_gtfs_time(st["departure_time"], now)
                if dep_dt is None:
                    continue

                # If time is past midnight (e.g. 25:00) it's next calendar day
                if dep_dt < now - timedelta(minutes=1):
                    dep_dt += timedelta(days=1)

                if dep_dt < now or dep_dt > cutoff:
                    continue

                trip    = self._trips.get(trip_id, {})
                route   = self._routes.get(trip.get("route_id", ""), {})
                mins    = int((dep_dt - now).total_seconds() / 60)

                departures.append({
                    "line":      route.get("short_name", "?"),
                    "direction": trip.get("headsign", ""),
                    "time":      dep_dt.strftime("%H:%M"),
                    "minutes":   mins,
                    "stop_id":   stop_id,
                })

            # Sort by departure time, deduplicate same line+direction within 1 min
            departures.sort(key=lambda d: d["minutes"])
            seen = set()
            deduped = []
            for d in departures:
                key = (d["line"], d["direction"], d["minutes"] // 2)
                if key not in seen:
                    seen.add(key)
                    deduped.append(d)

            result[group_key] = {
                "label":      group["label"],
                "icon":       group["icon"],
                "departures": deduped[:limit],
            }

        # Check if today has mass cancellations
        date_str = now.strftime("%Y%m%d")
        cancelled_sids = {
            sid for sid, dates in self._cal_dates.items()
            if dates.get(date_str) == "2"
        }
        added_sids = {
            sid for sid, dates in self._cal_dates.items()
            if dates.get(date_str) == "1"
        }
        all_cancelled = len(cancelled_sids) > 5 and len(added_sids) == 0

        return {
            "generated_at":   now.strftime("%H:%M:%S"),
            "cancelled_today": all_cancelled,
            "warning":        "Scheduled timetable shown — services may be altered today." if all_cancelled else None,
            "stops":          result,
        }