"""
Commerce Intelligence layer for Glam AI — now backed by REAL product data.

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

3. All product names, prices, and URLs below were verified via live
   web search at the time this file was written (August 2026) — prices
   on the real site WILL drift over time, sales end, etc. This is
   normal for any e-commerce integration; it's why real systems either
   re-fetch live or clearly show "last updated" timestamps. Consider
   mentioning this as a real-world design consideration in your report.

4. Every URL points to a REAL product page on nykaa.com. Clicking through
   takes the user to that actual product, where they can select the
   specific shade named here from the on-page shade selector.

5. RATING field is included where we found real, verified Nykaa rating
   data at research time (rating out of 5, with review count) — it's
   None where we didn't verify a rating, rather than guessed.

6. This catalog spans a genuine price range now (₹110 budget items up to
   ₹809 premium items) across multiple brand tiers — "best deal" reflects
   an actual cheapest-match search across brands, not just within one
   brand's range. It's still limited to the products we manually
   researched here, not the full real market — see the price-drift note
   in the UI for the same reason.
"""

import random

# ---------------------------------------------------------------------
# FOUNDATION CATALOG
# ---------------------------------------------------------------------

_FOUNDATION_PRODUCTS = [
    {
        "line": "Lakme 9 To 5 Powerplay Priming Foundation",
        "base_price": 489,
        "url": "https://www.nykaa.com/lakme-9-to-5-primer-matte-perfect-cover-foundation/p/574201",
        "rating": None,
        "shades": {
            "Warm": ["W120 Warm Creme", "W160 Warm Sand", "W180 Warm Natural",
                     "W240 Warm Beige", "W320 Warm Caramel", "Warm Light"],
            "Cool": ["C100 Cool Ivory", "C280 Cool Tan", "C300 Cool Cinnamon"],
            "Neutral": ["N200 Neutral Nude", "N220 Neutral Medium", "N260 Neutral Honey"],
        },
    },
    {
        "line": "Lakme 9to5 Hya Matte Foundation + Hyaluronic Acid",
        "base_price": 809,
        "url": "https://www.nykaa.com/lakme-9-to-5-hya-matte-foundation-hyaluronic-acid/p/19156410",
        "rating": None,
        "shades": {
            "Warm": ["Warm Creme", "Warm Light", "Warm Wood", "Warm Sand",
                     "Warm Natural", "Warm Beige"],
            "Cool": ["Cool Walnut", "Cool Cocoa", "Cool Mocha", "Cool Ivory",
                     "Cool Cinnamon", "Cool Rose", "Cool Tan"],
            "Neutral": ["Neutral Almond", "Neutral Chestnut", "Neutral Light",
                        "Neutral Nude", "Neutral Medium", "Neutral Honey"],
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
            "Warm": ["Nude Warm", "Beige Sand"],
            "Cool": ["Rose Ivory", "Fair Ivory"],
            "Neutral": ["Beige Natural", "Pale Medium"],
        },
    },
]

# ---------------------------------------------------------------------
# LIPSTICK CATALOG — undertone is our own color-family classification
# (see methodology note #2 above), prices/URLs are real
# ---------------------------------------------------------------------

_LIPSTICK_PRODUCTS = [
    {
        "line": "Lakme 9 To 5 Powerplay Priming Matte Lipstick",
        "base_price": 520,
        "url": "https://www.nykaa.com/lakme-9-to-5-primer-matte-lipstick/p/1037767",
        "rating": None,
        "shades": {
            "Warm": ["Caramel Latte", "Chocolate Crush", "Cinnamon Spice", "Red Twist",
                     "Peachy Affair", "Brown Walnut", "Rustic Brown", "Coffee Command",
                     "Scarlet Surge", "Brick Blush", "Iconic Red", "Red Velvet"],
            "Cool": ["Mauve Matter", "Burgundy Passion", "Blush Pink", "Dusty Pink",
                     "Cherry Chic"],
            "Neutral": ["Rose Day"],
        },
    },
    {
        "line": "Lakme 9to5 Hya Matte Liquid Lipstick",
        "base_price": 719,
        "url": "https://www.nykaa.com/lakme-9to5-hya-matte-liquid-lipstick/p/25044515",
        "rating": None,
        "shades": {
            "Warm": ["Peachy Pro"],
            "Cool": ["Pro Pink Medium", "After Hours Pink", "Werk Mauve",
                     "Werk Mauve Medium", "Monday Berry", "Pink On Point",
                     "All Day Pink", "Win Win Pink", "Pro Pink"],
            "Neutral": ["Werk Rose", "Nude"],
        },
    },
    {
        "line": "Maybelline New York Sensational Liquid Matte Lipstick",
        "base_price": 429,  # MRP — discount % fluctuates over time (seen 35% and 32% off on
                            # different checks), so MRP is the more stable reference point
        "url": "https://www.nykaa.com/maybelline-new-york-sensational-liquid-matte-lipstick/p/648684",
        "rating": {"stars": 4.3, "count": 293829},
        "shades": {
            "Warm": ["Flush It Red"],
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
            "Warm": ["Grateful Red"],
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
            "Warm": ["Code Red"],
            "Cool": ["Grape Riot", "Cherry Wine", "Berry Dance", "Mauve Date"],
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
            "Warm": ["Bloody Hell", "Coco", "Cherry Bomb", "Blood Lust", "Red Ocean", "Desert Taupe"],
            "Cool": ["Berries On Ice"],
            "Neutral": [],
        },
    },
]


def _build_catalog(products):
    """Flatten the product/shade structure into individual purchasable items."""
    catalog = {"Warm": [], "Cool": [], "Neutral": []}
    for product in products:
        for undertone, shade_list in product["shades"].items():
            for shade in shade_list:
                catalog[undertone].append({
                    "title": f"{product['line']} — {shade}",
                    "shade": shade,
                    "price": product["base_price"],
                    "platform": "Nykaa",
                    "url": product["url"],
                    "rating": product.get("rating"),
                })
    return catalog


FOUNDATION_CATALOG = _build_catalog(_FOUNDATION_PRODUCTS)
LIPSTICK_CATALOG = _build_catalog(_LIPSTICK_PRODUCTS)

_CATALOGS = {
    "lipstick": LIPSTICK_CATALOG,
    "foundation": FOUNDATION_CATALOG,
}


def find_best_deal(undertone: str, product_type: str = "lipstick") -> dict | None:
    """
    Returns the lowest-priced real product+shade match for the given
    undertone and product type, or None if nothing matches.
    """
    catalog = _CATALOGS.get(product_type)
    if not catalog:
        return None

    options = catalog.get(undertone)
    if not options:
        return None

    return min(options, key=lambda r: r["price"])


def find_all_deals(undertone: str, product_type: str = "lipstick") -> list:
    """Returns every real match for the given undertone/type, cheapest first."""
    catalog = _CATALOGS.get(product_type)
    if not catalog:
        return []
    options = catalog.get(undertone, [])
    return sorted(options, key=lambda r: r["price"])


if __name__ == "__main__":
    for ptype in ["lipstick", "foundation"]:
        print(f"\n=== {ptype.upper()} ===")
        for tone in ["Warm", "Cool", "Neutral"]:
            deal = find_best_deal(tone, ptype)
            count = len(_CATALOGS[ptype].get(tone, []))
            print(f"{tone} ({count} options) -> {deal}")