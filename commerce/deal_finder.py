"""
Commerce Intelligence layer for Glam AI — backed by a SQLite database
(products.db, auto-created from the seed data below on first run).

METHODOLOGY / HONESTY NOTES (read before presenting this in your demo/viva):

1. FOUNDATION undertones are OFFICIAL where the brand itself labels them in
   the shade name (Lakme does this consistently — e.g. "W120 Warm Creme").
   Swiss Beauty does NOT label most of its shades by undertone (only one
   shade, "Nude Warm", is explicit) — for the rest, the undertone tag is
   OUR OWN estimate based on the shade's described color family, same
   methodology as lipstick below. Say so if asked in your viva; don't
   claim Swiss Beauty shades are officially undertone-labeled.

2. LIPSTICK entries: brands generally do NOT label lipstick shades by
   undertone (they use names like "Cherry Chic" or "Rustic Brown"
   instead). The undertone tag on each lipstick below is MY OWN
   reasonable classification based on the shade's described color
   family (warm = red/orange/brown/coral/terracotta tones, cool =
   pink/berry/mauve/wine/purple tones, neutral = true nude/rose tones).
   Ambiguous/abstract shade names (e.g. "Boujee", "Wild Card") were
   deliberately left OUT rather than guessed, since there's no reliable
   way to infer color from a mood-based name alone.

3. All product names, prices, and Nykaa URLs below were verified via live
   web search at the time this file was written (August 2026) — prices
   on the real site WILL drift over time, sales end, etc. This is
   normal for any e-commerce integration; it's why real systems either
   re-fetch live or clearly show "last updated" timestamps. Consider
   mentioning this as a real-world design consideration in your report.
   The Amazon/Flipkart URLs added alongside them were found the same way,
   for real listings of the same named shade — where we couldn't find a
   given shade listed on a second platform, we simply didn't add one
   rather than invent a URL.

4. Every URL points to a REAL product page. Nykaa's/Amazon's/Flipkart's
   listings for the same shade generally carry the same brand-set MRP —
   we didn't find evidence of genuine per-platform MRP differences during
   research, so the same MRP is stored for each platform link on a given
   shade rather than a fabricated variance. The "sorted cheapest first"
   ordering still functions correctly; it just won't show a price
   difference for shades priced identically everywhere, which reflects
   real MRP-based pricing rather than a code limitation.

5. shade_rgb_r/g/b on every shade is an ESTIMATED RGB color inferred from
   the shade's name and description (e.g. "Iconic Red" -> a saturated
   red), NOT an official brand-provided color value or a value measured
   from a real product photo. It exists to power find_closest_shade()'s
   color-distance search — close enough for meaningful ranking, not
   pixel-perfect.

6. RATING field is included where we found real, verified Nykaa rating
   data at research time (rating out of 5, with review count) — it's
   None where we didn't verify a rating, rather than guessed.

7. This catalog spans a genuine price range now (₹110 budget items up to
   ₹809 premium items) across multiple brand tiers — "best deal" reflects
   an actual cheapest-match search across brands, not just within one
   brand's range. It's still limited to the products we manually
   researched here, not the full real market — see the price-drift note
   above for the same reason.
"""

import os
import sqlite3

DB_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "products.db"))

# ---------------------------------------------------------------------
# FOUNDATION SEED DATA
# shade_rgb is an estimated color per shade (see methodology note #5).
# extra_links adds Amazon/Flipkart alongside the default Nykaa link for
# shades we found listed on a second platform during research.
# ---------------------------------------------------------------------

_FOUNDATION_PRODUCTS = [
    {
        "line": "Lakme 9 To 5 Powerplay Priming Foundation",
        "base_price": 489,
        "url": "https://www.nykaa.com/lakme-9-to-5-primer-matte-perfect-cover-foundation/p/574201",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "W120 Warm Creme", "rgb": (225, 185, 150),
                 "extra_links": [{"platform": "Flipkart", "url": "https://www.flipkart.com/lakme-9to5-primer-matte-perfect-cover-foundation/p/itmf4f898d31ccac"}]},
                {"name": "W160 Warm Sand", "rgb": (215, 175, 135)},
                {"name": "W180 Warm Natural", "rgb": (205, 165, 125)},
                {"name": "W240 Warm Beige", "rgb": (195, 155, 115)},
                {"name": "W320 Warm Caramel", "rgb": (175, 130, 90)},
                {"name": "Warm Light", "rgb": (230, 195, 160)},
            ],
            "Cool": [
                {"name": "C100 Cool Ivory", "rgb": (230, 200, 190)},
                {"name": "C280 Cool Tan", "rgb": (180, 140, 140)},
                {"name": "C300 Cool Cinnamon", "rgb": (165, 120, 120)},
            ],
            "Neutral": [
                {"name": "N200 Neutral Nude", "rgb": (210, 175, 150)},
                {"name": "N220 Neutral Medium", "rgb": (195, 160, 135)},
                {"name": "N260 Neutral Honey", "rgb": (185, 145, 120)},
            ],
        },
    },
    {
        "line": "Lakme 9to5 Hya Matte Foundation + Hyaluronic Acid",
        "base_price": 809,
        "url": "https://www.nykaa.com/lakme-9-to-5-hya-matte-foundation-hyaluronic-acid/p/19156410",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "Warm Creme", "rgb": (225, 185, 150)},
                {"name": "Warm Light", "rgb": (232, 197, 163)},
                {"name": "Warm Wood", "rgb": (150, 105, 75)},
                {"name": "Warm Sand", "rgb": (215, 175, 135)},
                {"name": "Warm Natural", "rgb": (205, 165, 125)},
                {"name": "Warm Beige", "rgb": (195, 155, 115)},
            ],
            "Cool": [
                {"name": "Cool Walnut", "rgb": (140, 105, 105)},
                {"name": "Cool Cocoa", "rgb": (120, 85, 85)},
                {"name": "Cool Mocha", "rgb": (150, 110, 115)},
                {"name": "Cool Ivory", "rgb": (230, 200, 195)},
                {"name": "Cool Cinnamon", "rgb": (165, 120, 125)},
                {"name": "Cool Rose", "rgb": (210, 170, 175)},
                {"name": "Cool Tan", "rgb": (180, 140, 145)},
            ],
            "Neutral": [
                {"name": "Neutral Almond", "rgb": (190, 150, 130)},
                {"name": "Neutral Chestnut", "rgb": (160, 115, 95)},
                {"name": "Neutral Light", "rgb": (225, 190, 165)},
                {"name": "Neutral Nude", "rgb": (210, 175, 150)},
                {"name": "Neutral Medium", "rgb": (195, 160, 140)},
                {"name": "Neutral Honey", "rgb": (185, 145, 120)},
            ],
        },
    },
    {
        # Budget option — undertones are OUR classification except "Nude Warm"
        # which the brand itself labels
        "line": "Swiss Beauty Flawless Complexion Foundation",
        "base_price": 229,
        "url": "https://www.nykaa.com/swiss-beauty-flawless-complexion-foundation/p/4714174",
        "rating": {"stars": 4.1, "count": 398},
        "shades": {
            "Warm": [
                {"name": "Nude Warm", "rgb": (215, 175, 140)},
                {"name": "Beige Sand", "rgb": (205, 165, 130)},
            ],
            "Cool": [
                {"name": "Rose Ivory", "rgb": (225, 190, 190)},
                {"name": "Fair Ivory", "rgb": (235, 205, 195)},
            ],
            "Neutral": [
                {"name": "Beige Natural", "rgb": (200, 165, 145)},
                {"name": "Pale Medium", "rgb": (215, 180, 160)},
            ],
        },
    },
]

# ---------------------------------------------------------------------
# LIPSTICK SEED DATA — undertone is our own color-family classification
# (see methodology note #2 above), prices/Nykaa URLs are real
# ---------------------------------------------------------------------

_LIPSTICK_PRODUCTS = [
    {
        "line": "Lakme 9 To 5 Powerplay Priming Matte Lipstick",
        "base_price": 520,
        "url": "https://www.nykaa.com/lakme-9-to-5-primer-matte-lipstick/p/1037767",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "Caramel Latte", "rgb": (180, 120, 80)},
                {"name": "Chocolate Crush", "rgb": (110, 60, 45)},
                {"name": "Cinnamon Spice", "rgb": (165, 75, 55)},
                {"name": "Red Twist", "rgb": (190, 40, 45),
                 "extra_links": [{"platform": "Flipkart", "url": "https://www.flipkart.com/lakm-9to5-primer-matte-lip-color-red-twist/p/itmd72a593703d1c"}]},
                {"name": "Peachy Affair", "rgb": (230, 130, 100)},
                {"name": "Brown Walnut", "rgb": (120, 70, 55)},
                {"name": "Rustic Brown", "rgb": (140, 80, 60)},
                {"name": "Coffee Command", "rgb": (100, 60, 50)},
                {"name": "Scarlet Surge", "rgb": (200, 30, 35)},
                {"name": "Brick Blush", "rgb": (175, 85, 65)},
                {"name": "Iconic Red", "rgb": (180, 30, 40),
                 "extra_links": [{"platform": "Amazon", "url": "https://www.amazon.in/Lakme-Lipstick-Iconic-Red-Matte/dp/B09263NV18"}]},
                {"name": "Red Velvet", "rgb": (150, 25, 35),
                 "extra_links": [
                     {"platform": "Amazon", "url": "https://www.amazon.in/Lakme-Lipstick-Red-Velvet-Matte/dp/B09264KR7K"},
                     {"platform": "Flipkart", "url": "https://www.flipkart.com/lakm-9to5-primer-matte-lip-color-red-velvet/p/itm631c57ea58ea4"},
                 ]},
            ],
            "Cool": [
                {"name": "Mauve Matter", "rgb": (150, 90, 100)},
                {"name": "Burgundy Passion", "rgb": (110, 30, 45)},
                {"name": "Blush Pink", "rgb": (220, 150, 160),
                 "extra_links": [{"platform": "Amazon", "url": "https://www.amazon.in/Lakme-Lipstick-Blush-Pink-Matte/dp/B09262ZT8F"}]},
                {"name": "Dusty Pink", "rgb": (200, 130, 140)},
                {"name": "Cherry Chic", "rgb": (180, 40, 70)},
            ],
            "Neutral": [
                {"name": "Rose Day", "rgb": (200, 120, 120)},
            ],
        },
    },
    {
        "line": "Lakme 9to5 Hya Matte Liquid Lipstick",
        "base_price": 719,
        "url": "https://www.nykaa.com/lakme-9to5-hya-matte-liquid-lipstick/p/25044515",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "Peachy Pro", "rgb": (225, 140, 110)},
            ],
            "Cool": [
                {"name": "Pro Pink Medium", "rgb": (220, 120, 140)},
                {"name": "After Hours Pink", "rgb": (200, 60, 100)},
                {"name": "Werk Mauve", "rgb": (160, 100, 110)},
                {"name": "Werk Mauve Medium", "rgb": (170, 110, 120)},
                {"name": "Monday Berry", "rgb": (150, 50, 80)},
                {"name": "Pink On Point", "rgb": (230, 100, 140)},
                {"name": "All Day Pink", "rgb": (225, 130, 150)},
                {"name": "Win Win Pink", "rgb": (215, 110, 135)},
                {"name": "Pro Pink", "rgb": (220, 120, 145)},
            ],
            "Neutral": [
                {"name": "Werk Rose", "rgb": (195, 130, 125)},
                {"name": "Nude", "rgb": (210, 160, 140),
                 "extra_links": [{"platform": "Amazon", "url": "https://www.amazon.in/Lipstick-Hyaluronic-Vitamin-Smudge-Proof-Long-Lasting/dp/B0GG9LSF1Q"}]},
            ],
        },
    },
    {
        "line": "Maybelline New York Sensational Liquid Matte Lipstick",
        "base_price": 429,  # MRP — discount % fluctuates over time (seen 35% and 32% off on
                            # different checks), so MRP is the more stable reference point
        "url": "https://www.nykaa.com/maybelline-new-york-sensational-liquid-matte-lipstick/p/648684",
        "rating": {"stars": 4.3, "count": 293829},
        "shades": {
            "Warm": [
                {"name": "Flush It Red", "rgb": (200, 40, 35),
                 "extra_links": [
                     {"platform": "Amazon", "url": "https://www.amazon.in/Maybelline-Sensational-Liquid-Matte-Lipstick/dp/B08JWCKS2S"},
                     {"platform": "Flipkart", "url": "https://www.flipkart.com/maybelline-new-york-sensational-liquid-matte-lipstick-03-flush-red-11-made-easy/p/itm6f517c4eb3464"},
                 ]},
            ],
            "Cool": [],
            "Neutral": [],
        },
    },
    {
        "line": "SUGAR Matte Attack Transferproof Lipstick",
        "base_price": 562,
        "url": "https://www.nykaa.com/sugar-cosmetics-matte-attack-transferproof-lipstick/p/648580",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "Grateful Red", "rgb": (190, 35, 40),
                 "extra_links": [
                     {"platform": "Amazon", "url": "https://www.amazon.in/SUGAR-Matte-Attack-Transferproof-Lipstick/dp/B08WHY7J57"},
                     {"platform": "Flipkart", "url": "https://www.flipkart.com/sugar-cosmetics-matte-attack-transferproof-lipstick-17-grateful-red/p/itm6544e2c832064"},
                 ]},
            ],
            "Cool": [],
            "Neutral": [],
        },
    },
    {
        # Budget option — genuinely one of the cheapest lipsticks on Nykaa
        "line": "Elle 18 Color Pop Matte Lip Color",
        "base_price": 110,
        "url": "https://www.nykaa.com/elle-18-color-pop-matte-lip-color/p/2732715",
        "rating": None,
        "shades": {
            "Warm": [
                {"name": "Code Red", "rgb": (195, 35, 35),
                 "extra_links": [{"platform": "Amazon", "url": "https://www.amazon.in/Elle18-Color-Pops-Matte-Code/dp/B08D6P8KNF"}]},
            ],
            "Cool": [
                {"name": "Grape Riot", "rgb": (110, 50, 90)},
                {"name": "Cherry Wine", "rgb": (120, 30, 50),
                 "extra_links": [{"platform": "Flipkart", "url": "https://www.flipkart.com/elle-18-color-pop-matte-lip-deep-pink-maroon-silk-misty-magenta-cherry-wine-grape-riot-coral-dose-4-3-g/p/itmcfa595eba71b0"}]},
                {"name": "Berry Dance", "rgb": (150, 50, 90)},
                {"name": "Mauve Date", "rgb": (160, 100, 120),
                 "extra_links": [{"platform": "Flipkart", "url": "https://www.flipkart.com/elle-18-color-pops-matte-lipstick/p/itm3684269c15b54"}]},
            ],
            "Neutral": [],
        },
    },
    {
        # Budget option with a real, strong rating
        "line": "Insight Cosmetics Matte Lip Ink",
        "base_price": 170,
        "url": "https://www.nykaa.com/insight-cosmetics-matte-lip-ink/p/2641078",
        "rating": {"stars": 4.3, "count": 25804},
        "shades": {
            "Warm": [
                {"name": "Bloody Hell", "rgb": (150, 20, 25)},
                {"name": "Coco", "rgb": (130, 70, 55)},
                {"name": "Cherry Bomb", "rgb": (180, 30, 45)},
                {"name": "Blood Lust", "rgb": (140, 15, 20)},
                {"name": "Red Ocean", "rgb": (170, 35, 40)},
                {"name": "Desert Taupe", "rgb": (170, 120, 100)},
            ],
            "Cool": [
                {"name": "Berries On Ice", "rgb": (170, 90, 110)},
            ],
            "Neutral": [],
        },
    },
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_name TEXT NOT NULL,
    product_type TEXT NOT NULL CHECK(product_type IN ('lipstick', 'foundation'))
);

CREATE TABLE IF NOT EXISTS shades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES products(id),
    shade_name TEXT NOT NULL,
    undertone TEXT NOT NULL CHECK(undertone IN ('Warm', 'Cool', 'Neutral')),
    shade_rgb_r INTEGER NOT NULL,
    shade_rgb_g INTEGER NOT NULL,
    shade_rgb_b INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS platform_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shade_id INTEGER NOT NULL REFERENCES shades(id),
    platform_name TEXT NOT NULL CHECK(platform_name IN ('Nykaa', 'Amazon', 'Flipkart')),
    price INTEGER NOT NULL,
    url TEXT NOT NULL,
    rating_stars REAL,
    rating_count INTEGER
);
"""


def _seed_database(conn):
    """Populates a freshly-created products.db from the hardcoded catalog
    above. Only ever runs once, against a brand-new database file — see
    _ensure_db()."""
    for product_type, products in (("foundation", _FOUNDATION_PRODUCTS), ("lipstick", _LIPSTICK_PRODUCTS)):
        for product in products:
            rating = product.get("rating") or {}
            rating_stars = rating.get("stars")
            rating_count = rating.get("count")

            cur = conn.execute(
                "INSERT INTO products (line_name, product_type) VALUES (?, ?)",
                (product["line"], product_type),
            )
            product_id = cur.lastrowid

            for undertone, shade_list in product["shades"].items():
                for shade in shade_list:
                    r, g, b = shade["rgb"]
                    cur = conn.execute(
                        "INSERT INTO shades (product_id, shade_name, undertone, shade_rgb_r, shade_rgb_g, shade_rgb_b) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (product_id, shade["name"], undertone, r, g, b),
                    )
                    shade_id = cur.lastrowid

                    conn.execute(
                        "INSERT INTO platform_links (shade_id, platform_name, price, url, rating_stars, rating_count) "
                        "VALUES (?, 'Nykaa', ?, ?, ?, ?)",
                        (shade_id, product["base_price"], product["url"], rating_stars, rating_count),
                    )
                    for link in shade.get("extra_links", []):
                        conn.execute(
                            "INSERT INTO platform_links (shade_id, platform_name, price, url, rating_stars, rating_count) "
                            "VALUES (?, ?, ?, ?, NULL, NULL)",
                            (shade_id, link["platform"], product["base_price"], link["url"]),
                        )


def _ensure_db():
    """Auto-creates and seeds products.db on first run — no manual setup
    step needed. Never touches an existing database file."""
    if os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(_SCHEMA)
        _seed_database(conn)
        conn.commit()
    finally:
        conn.close()


def _get_connection():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_deal(row) -> dict:
    rating = None
    if row["rating_stars"] is not None:
        rating = {"stars": row["rating_stars"], "count": row["rating_count"]}
    return {
        "title": f"{row['line_name']} — {row['shade_name']}",
        "shade": row["shade_name"],
        "price": row["price"],
        "platform": row["platform_name"],
        "url": row["url"],
        "rating": rating,
    }


def find_best_deal(undertone: str, product_type: str = "lipstick") -> dict | None:
    """
    Returns the lowest-priced real product+shade match for the given
    undertone and product type, or None if nothing matches.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT s.shade_name, p.line_name, pl.price, pl.url, pl.platform_name,
                   pl.rating_stars, pl.rating_count
            FROM shades s
            JOIN products p ON p.id = s.product_id
            JOIN platform_links pl ON pl.shade_id = s.id
            WHERE p.product_type = ? AND s.undertone = ?
            ORDER BY pl.price ASC
            LIMIT 1
            """,
            (product_type, undertone),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return _row_to_deal(row)


def find_all_deals(undertone: str, product_type: str = "lipstick") -> list:
    """Returns every real shade match for the given undertone/type — one
    row per shade (its cheapest platform link; ties broken deterministically
    by link id so shades with multiple same-priced platform links don't get
    double-counted), cheapest first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT s.shade_name, p.line_name, pl.price, pl.url, pl.platform_name,
                   pl.rating_stars, pl.rating_count
            FROM shades s
            JOIN products p ON p.id = s.product_id
            JOIN platform_links pl ON pl.id = (
                SELECT id FROM platform_links WHERE shade_id = s.id ORDER BY price ASC, id ASC LIMIT 1
            )
            WHERE p.product_type = ? AND s.undertone = ?
            ORDER BY pl.price ASC
            """,
            (product_type, undertone),
        ).fetchall()
    finally:
        conn.close()

    return [_row_to_deal(row) for row in rows]


def find_closest_shade(product_rgb, product_type: str = "lipstick") -> dict | None:
    """
    Given an extracted product color (from analysis.product_match's
    extract_product_color), finds the catalog shade of the given type
    whose estimated color is closest in CIE Lab space (reusing
    color_distance from analysis/product_match.py rather than
    reimplementing it), and returns that shade with ALL its platform
    links, cheapest first. Returns None if the catalog has no shades of
    that product type.
    """
    from analysis.product_match import color_distance

    conn = _get_connection()
    try:
        shades = conn.execute(
            """
            SELECT s.id as shade_id, s.shade_name, s.shade_rgb_r, s.shade_rgb_g, s.shade_rgb_b,
                   p.line_name
            FROM shades s
            JOIN products p ON p.id = s.product_id
            WHERE p.product_type = ?
            """,
            (product_type,),
        ).fetchall()

        if not shades:
            return None

        best_shade = None
        best_distance = None
        for shade in shades:
            shade_rgb = (shade["shade_rgb_r"], shade["shade_rgb_g"], shade["shade_rgb_b"])
            distance = color_distance(product_rgb, shade_rgb)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_shade = shade

        links = conn.execute(
            """
            SELECT platform_name, price, url, rating_stars, rating_count
            FROM platform_links
            WHERE shade_id = ?
            ORDER BY price ASC
            """,
            (best_shade["shade_id"],),
        ).fetchall()
    finally:
        conn.close()

    r, g, b = best_shade["shade_rgb_r"], best_shade["shade_rgb_g"], best_shade["shade_rgb_b"]
    shade_hex = '#%02x%02x%02x' % (r, g, b)

    return {
        "shade_name": best_shade["shade_name"],
        "line_name": best_shade["line_name"],
        "match_label": "Exact match" if best_distance < 10 else "Closest match found",
        "distance": round(best_distance, 1),
        "shade_hex": shade_hex,
        "platform_links": [
            {
                "platform": link["platform_name"],
                "price": link["price"],
                "url": link["url"],
                "rating": {"stars": link["rating_stars"], "count": link["rating_count"]} if link["rating_stars"] is not None else None,
            }
            for link in links
        ],
    }


if __name__ == "__main__":
    for ptype in ["lipstick", "foundation"]:
        print(f"\n=== {ptype.upper()} ===")
        for tone in ["Warm", "Cool", "Neutral"]:
            deal = find_best_deal(tone, ptype)
            all_deals = find_all_deals(tone, ptype)
            print(f"{tone} ({len(all_deals)} shades) -> {deal}")
