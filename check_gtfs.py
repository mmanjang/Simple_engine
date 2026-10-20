"""
check_gtfs.py
─────────────
Reads your GTFS zip and prints:
  - The feed's valid date range (from feed_info.txt or calendar.txt)
  - A sample of routes (bus/tram/train lines)
  - A sample of stops

Run this before using PT routing to confirm what dates and services
are available in your feed.

Usage:  python check_gtfs.py
        python check_gtfs.py --file data/my_feed.zip
"""

import zipfile
import csv
import io
import argparse
import os
import glob


def find_gtfs_file() -> str:
    """Auto-detect the GTFS zip in the data/ folder."""
    candidates = glob.glob("data/magdeburg_gtfs_new.zip")
    if not candidates:
        raise FileNotFoundError(
            "No .zip file found in data/. "
            "Pass the path explicitly: python check_gtfs.py --file data/yourfeed.zip"
        )
    if len(candidates) == 1:
        return candidates[0]
    print(f"Multiple zip files found: {candidates}")
    return candidates[0]


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> list[dict]:
    """Read a CSV file from inside the zip and return list of row dicts."""
    try:
        with zf.open(filename) as f:
            content = f.read().decode("utf-8-sig")  # utf-8-sig strips BOM
            reader  = csv.DictReader(io.StringIO(content))
            return list(reader)
    except KeyError:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", dest="file", default=None,
                        help="Path to GTFS zip file (auto-detected if not given)")
    args = parser.parse_args()

    gtfs_path = args.file or find_gtfs_file()

    if not os.path.exists(gtfs_path):
        print(f"❌ File not found: {gtfs_path}")
        return

    print(f"\n📦 GTFS Feed Inspector")
    print(f"   File: {gtfs_path}")
    print("="*62)

    with zipfile.ZipFile(gtfs_path, "r") as zf:

        files_in_zip = [e.filename for e in zf.infolist()]
        print(f"\n📁 Files in feed: {', '.join(files_in_zip)}")

        # ── Date range ────────────────────────────────────────────────────────
        print("\n📅 VALID DATE RANGE")
        print("   ─────────────────────────────────────────────────────")

        feed_info = read_csv_from_zip(zf, "feed_info.txt")
        if feed_info:
            row = feed_info[0]
            start = row.get("feed_start_date", "?")
            end   = row.get("feed_end_date",   "?")
            print(f"   From feed_info.txt:")
            print(f"   Start : {start}")
            print(f"   End   : {end}")
            print(f"\n   ✅ Use a departure date between {start} and {end}")
        else:
            # Fall back to calendar.txt
            calendar = read_csv_from_zip(zf, "calendar.txt")
            if calendar:
                starts = [r.get("start_date","") for r in calendar if r.get("start_date")]
                ends   = [r.get("end_date","")   for r in calendar if r.get("end_date")]
                if starts and ends:
                    print(f"   From calendar.txt (across {len(calendar)} service(s)):")
                    print(f"   Earliest start : {min(starts)}")
                    print(f"   Latest end     : {max(ends)}")
                    print(f"\n   ✅ Use a departure date between {min(starts)} and {max(ends)}")
            else:
                calendar_dates = read_csv_from_zip(zf, "calendar_dates.txt")
                if calendar_dates:
                    dates = sorted(set(r.get("date","") for r in calendar_dates if r.get("date")))
                    print(f"   From calendar_dates.txt ({len(dates)} unique dates):")
                    print(f"   Earliest : {dates[0]}")
                    print(f"   Latest   : {dates[-1]}")
                    print(f"\n   ✅ Use a departure date between {dates[0]} and {dates[-1]}")
                else:
                    print("   ⚠️  Could not determine date range (no feed_info, calendar, or calendar_dates found)")

        # ── Routes (lines) ────────────────────────────────────────────────────
        print("\n🚌 ROUTES (first 20)")
        print("   ─────────────────────────────────────────────────────")
        routes = read_csv_from_zip(zf, "routes.txt")
        if routes:
            route_type_names = {
                "0": "Tram/LRT",  "1": "Metro",    "2": "Rail",
                "3": "Bus",       "4": "Ferry",     "5": "Cable car",
                "6": "Gondola",   "7": "Funicular", "11": "Trolleybus",
                "12": "Monorail",
            }
            print(f"   Total routes: {len(routes)}")
            print()
            print(f"   {'ID':<15} {'SHORT NAME':<12} {'LONG NAME':<30} TYPE")
            print(f"   {'─'*70}")
            for r in routes[:20]:
                rtype = route_type_names.get(r.get("route_type",""), f"type {r.get('route_type','?')}")
                short = r.get("route_short_name", "")
                long  = r.get("route_long_name",  "")[:30]
                rid   = r.get("route_id", "")[:15]
                print(f"   {rid:<15} {short:<12} {long:<30} {rtype}")
            if len(routes) > 20:
                print(f"   ... and {len(routes)-20} more routes")
        else:
            print("   ⚠️  No routes.txt found in feed")

        # ── Stops ─────────────────────────────────────────────────────────────
        print("\n🚏 STOPS (first 15)")
        print("   ─────────────────────────────────────────────────────")
        stops = read_csv_from_zip(zf, "stops.txt")
        if stops:
            print(f"   Total stops: {len(stops)}")
            print()
            print(f"   {'NAME':<35} {'LAT':<12} {'LON'}")
            print(f"   {'─'*62}")
            for s in stops[:15]:
                name = s.get("stop_name", "?")[:35]
                lat  = s.get("stop_lat",  "?")
                lon  = s.get("stop_lon",  "?")
                print(f"   {name:<35} {lat:<12} {lon}")
            if len(stops) > 15:
                print(f"   ... and {len(stops)-15} more stops")
        else:
            print("   ⚠️  No stops.txt found in feed")

    print("\n" + "="*62)
    print("  💡 To use PT routing with this feed, run:")
    print('     python main.py --departure "YYYY-MM-DDTHH:MM:SS+HH:MM"')
    print("  Replace YYYY-MM-DD with a date in the valid range above.")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
