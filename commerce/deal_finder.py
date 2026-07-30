"""
Commerce Intelligence layer for Glam AI — now backed by REAL product data.

METHODOLOGY / HONESTY NOTES (read before presenting this in your demo/viva):

1. FOUNDATION entries: undertone labels (Warm/Cool/Neutral) are OFFICIAL —
   they're taken directly from the shade names Lakme themselves use
   (e.g. "W120 Warm Creme", "C280 Cool Tan", "N200 Neutral Nude"). These
   are 100% as accurate as the brand's own labeling.

2. LIPSTICK entries: brands generally do NOT label lipstick shades by
   undertone (they use names like "Cherry Chic" or "Rustic Brown"
   instead). The undertone tag on each lipstick below is MY OWN
   reasonable classification based on the shade's described color
   family (warm = red/orange/brown/coral/terracotta tones, cool =
   pink/berry/mauve/wine/purple tones, neutral = true nude/rose tones).
   This is a judgment call, not an official brand claim — say so if
   asked in your viva.

3. All product names, prices, and URLs below were verified via live
   web search at the time this file was written (August 2026) — prices
   on the real site WILL drift over time, sales end, etc. This is
   normal for any e-commerce integration; it's why real systems either
   re-fetch live or clearly show "last updated" timestamps. Consider
   mentioning this as a real-world design consideration in your report.

4. Every URL points to a REAL product page on nykaa.com. Clicking through
   takes the user to that actual product, where they can select the
   specific shade named here from the on-page shade selector.
"""

import random

# ---------------------------------------------------------------------
# FOUNDATION CATALOG — undertones are Lakme's own official shade labels
# ---------------------------------------------------------------------

_FOUNDATION_PRODUCTS = [
    {
        "line": "Lakme 9 To 5 Powerplay Priming Foundation",
        "base_price": 489,
        "url": "https://www.nykaa.com/lakme-9-to-5-primer-matte-perfect-cover-foundation/p/574201",
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
        "shades": {
            "Warm": ["Warm Creme", "Warm Light", "Warm Wood", "Warm Sand",
                     "Warm Natural", "Warm Beige"],
            "Cool": ["Cool Walnut", "Cool Cocoa", "Cool Mocha", "Cool Ivory",
                     "Cool Cinnamon", "Cool Rose", "Cool Tan"],
            "Neutral": ["Neutral Almond", "Neutral Chestnut", "Neutral Light",
                        "Neutral Nude", "Neutral Medium", "Neutral Honey"],
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
        "base_price": 279,
        "url": "https://www.nykaa.com/maybelline-new-york-sensational-liquid-matte-lipstick/p/648684",
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
        "shades": {
            "Warm": ["Grateful Red"],
            "Cool": [],
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
                    "price": product["base_price"],
                    "platform": "Nykaa",
                    "url": product["url"],
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