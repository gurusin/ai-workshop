import os
import requests
from langchain_core.tools import tool


def _odds_key() -> str:
    return os.environ.get("ODDS_API_KEY", "")


def _weather_key() -> str:
    return os.environ.get("WEATHER_API_KEY", "")


@tool
def get_live_odds(sport_key: str) -> str:
    """
    Fetch live odds from multiple bookmakers for a sport.
    sport_key examples: americanfootball_nfl, basketball_nba, baseball_mlb,
    icehockey_nhl, soccer_epl, aussierules_afl, americanfootball_ncaaf.
    Returns moneyline, spread, and totals markets with cross-bookmaker line discrepancies flagged.
    """
    api_key = _odds_key()
    if not api_key:
        return (
            "ODDS_API_KEY not configured. "
            "Get a free key at https://the-odds-api.com and enter it in the sidebar."
        )

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us,uk,au",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "unknown")

        if resp.status_code == 401:
            return "ERROR: Invalid Odds API key."
        if resp.status_code == 422:
            return f"Sport key '{sport_key}' not found or not active. Try a different sport."
        if resp.status_code != 200:
            return f"Odds API error HTTP {resp.status_code}: {resp.text[:200]}"

        events = resp.json()
        if not events:
            return f"No live events found for '{sport_key}' right now (off-season or no scheduled games)."

        lines = [f"=== Live odds: {sport_key} | {len(events)} events | {remaining} API requests remaining ===\n"]

        for event in events[:12]:
            home = event.get("home_team", "?")
            away = event.get("away_team", "?")
            date = event.get("commence_time", "")[:10]
            bookmakers = event.get("bookmakers", [])

            lines.append(f"\n--- {away} @ {home}  ({date}) ---")

            home_spreads: dict[str, float] = {}
            over_totals: dict[str, float] = {}
            home_ml: dict[str, int] = {}

            for bk in bookmakers:
                bk_name = bk.get("title", "?")
                for mkt in bk.get("markets", []):
                    key = mkt.get("key")
                    outcomes = mkt.get("outcomes", [])
                    if key == "spreads":
                        for o in outcomes:
                            if o.get("name") == home:
                                home_spreads[bk_name] = float(o.get("point", 0))
                    elif key == "totals":
                        for o in outcomes:
                            if o.get("name") == "Over":
                                over_totals[bk_name] = float(o.get("point", 0))
                    elif key == "h2h":
                        for o in outcomes:
                            if o.get("name") == home:
                                home_ml[bk_name] = int(o.get("price", 0))

            if home_spreads:
                vals = list(home_spreads.values())
                consensus = sum(vals) / len(vals)
                discrepancy = max(vals) - min(vals)
                flag = "  ⚠️ LINE DISCREPANCY (sharp money signal)" if discrepancy >= 1.5 else ""
                spread_detail = " | ".join(f"{k}: {v:+.1f}" for k, v in list(home_spreads.items())[:6])
                lines.append(f"  Spread (home): {spread_detail}{flag}")
                lines.append(f"  Consensus: {home} {consensus:+.1f}  [range: {min(vals):+.1f} to {max(vals):+.1f}]")

            if over_totals:
                vals = list(over_totals.values())
                discrepancy = max(vals) - min(vals)
                flag = "  ⚠️ TOTAL DISCREPANCY" if discrepancy >= 1.5 else ""
                total_detail = " | ".join(f"{k}: {v}" for k, v in list(over_totals.items())[:6])
                lines.append(f"  Total (O/U): {total_detail}{flag}")

            if home_ml:
                ml_detail = " | ".join(f"{k}: {v:+d}" for k, v in list(home_ml.items())[:6])
                lines.append(f"  Moneyline (home): {ml_detail}")

        return "\n".join(lines)

    except requests.Timeout:
        return "ERROR: Odds API request timed out (15s). Check your connection."
    except Exception as exc:
        return f"ERROR fetching odds: {exc}"


@tool
def get_sports_news(league: str) -> str:
    """
    Fetch recent sports news and injury reports from ESPN's public API (no key required).
    league examples: nfl, nba, mlb, nhl, epl, afl, nrl.
    Injury items are automatically flagged.
    """
    LEAGUE_MAP = {
        "nfl":    ("football",             "nfl"),
        "nba":    ("basketball",           "nba"),
        "mlb":    ("baseball",             "mlb"),
        "nhl":    ("hockey",               "nhl"),
        "epl":    ("soccer",               "eng.1"),
        "afl":    ("australian-football",  "afl"),
        "nrl":    ("rugby-league",         "nrl"),
        "soccer": ("soccer",               "eng.1"),
    }

    sport_path, league_path = LEAGUE_MAP.get(league.lower(), ("football", "nfl"))
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/{league_path}/news"

    try:
        resp = requests.get(url, params={"limit": 15}, timeout=12)
        if resp.status_code != 200:
            return (
                f"ESPN news unavailable for '{league}' (HTTP {resp.status_code}). "
                "League may not be covered by ESPN."
            )

        articles = resp.json().get("articles", [])
        if not articles:
            return f"No recent articles found for {league.upper()}."

        INJURY_WORDS = {"injury", "injured", "out", "questionable", "doubtful", "scratch", "ir "}

        lines = [f"=== {league.upper()} News ({len(articles)} articles) ===\n"]
        for i, a in enumerate(articles[:12]):
            title = a.get("headline", "")
            desc = a.get("description", "")
            pub = a.get("published", "")[:10]

            is_injury = any(w in (title + " " + desc).lower() for w in INJURY_WORDS)
            prefix = "🏥 INJURY:" if is_injury else f"{i + 1}."
            lines.append(f"{prefix} {title} ({pub})")
            if desc:
                lines.append(f"   {desc[:220]}")

        return "\n".join(lines)

    except requests.Timeout:
        return f"ESPN news request timed out for {league}."
    except Exception as exc:
        return f"ERROR fetching sports news: {exc}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for an outdoor sports venue city.
    Useful for NFL, MLB, AFL, soccer games played in open stadiums.
    city examples: 'Chicago', 'Green Bay', 'London', 'Melbourne', 'Buffalo'
    """
    api_key = _weather_key()
    if not api_key:
        return (
            "WEATHER_API_KEY not configured (optional). "
            "Get a free key at https://openweathermap.org/api"
        )

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        if resp.status_code == 404:
            return f"City '{city}' not found. Try a different name."
        if resp.status_code != 200:
            return f"Weather API error HTTP {resp.status_code}."

        d = resp.json()
        temp_c = d["main"]["temp"]
        temp_f = temp_c * 9 / 5 + 32
        wind_ms = d["wind"]["speed"]
        wind_mph = wind_ms * 2.237
        description = d["weather"][0]["description"]
        humidity = d["main"]["humidity"]

        flags = []
        if wind_mph > 15:
            flags.append(f"WIND {wind_mph:.0f} mph — reduces passing efficiency and field-goal accuracy")
        if wind_mph > 25:
            flags.append("EXTREME WIND — strong lean toward the under")
        if temp_f < 32:
            flags.append(f"FREEZING {temp_f:.0f}°F — favours running game and under")
        if "rain" in description or "drizzle" in description:
            flags.append("RAIN — slippery ball, leans toward under and run-heavy teams")
        if "snow" in description:
            flags.append("SNOW — significant under lean, favours stronger running team")

        parts = [
            f"=== Weather in {d['name']} ===",
            f"Conditions: {description}",
            f"Temperature: {temp_c:.1f}°C / {temp_f:.1f}°F",
            f"Wind: {wind_mph:.0f} mph ({wind_ms * 3.6:.0f} km/h)",
            f"Humidity: {humidity}%",
        ]
        if flags:
            parts.append("\nBetting impact:")
            parts += [f"  ⚠️ {f}" for f in flags]
        else:
            parts.append("No significant weather betting impact expected.")

        return "\n".join(parts)

    except requests.Timeout:
        return f"Weather request timed out for '{city}'."
    except Exception as exc:
        return f"ERROR fetching weather: {exc}"
