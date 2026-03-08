import pandas as pd
from datasets import load_dataset


def load_training_data():
    print("Loading transaction data from Hugging Face...")
    try:
        dataset = load_dataset("mitulshah/transaction-categorization", split="train[:5000]")
        df = dataset.to_pandas()
        df.columns = df.columns.str.strip().str.lower()

        desc_col = next((c for c in df.columns if 'desc' in c or 'transaction' in c or 'name' in c), None)
        cat_col  = next((c for c in df.columns if 'cat' in c or 'label' in c or 'type' in c), None)

        if not desc_col or not cat_col:
            raise ValueError(f"Could not detect columns. Available: {list(df.columns)}")

        df = df.rename(columns={desc_col: 'description', cat_col: 'category'})
        df = df[['description', 'category']].dropna()
        df['description'] = df['description'].astype(str).str.lower().str.strip()

        print(f"Loaded {len(df)} rows across {df['category'].nunique()} categories")
        return df

    except Exception as e:
        print(f"Hugging Face load failed: {e}. Falling back to built-in data...")
        return _fallback()


def _fallback():
    templates = {
        "Food & Dining": [
            "mcdonalds", "burger king", "wendys", "taco bell", "subway", "chipotle",
            "starbucks", "dunkin donuts", "dominos", "pizza hut", "panera bread",
            "chick fil a", "five guys", "shake shack", "grubhub", "uber eats",
            "doordash", "whole foods", "trader joes", "popeyes", "kfc",
            "panda express", "olive garden", "applebees", "dennys", "ihop",
            "waffle house", "sonic drive in", "dairy queen", "little caesars",
            "papa johns", "jimmy johns", "jersey mikes", "wingstop",
            "grubhub order", "uber eats delivery", "doordash order",
            "instacart groceries", "walmart groceries", "costco groceries"
        ],
        "Transportation": [
            "uber", "lyft", "grab", "shell gas", "bp fuel", "exxon gas",
            "chevron station", "delta airlines", "united airlines", "american airlines",
            "southwest airlines", "amtrak", "greyhound bus", "mta subway",
            "parking garage", "hertz rental", "enterprise car", "zipcar",
            "metro card", "toll payment", "avis car", "budget rental",
            "spirit airlines", "jetblue", "frontier airlines", "uber trip",
            "lyft ride", "taxi cab", "via ride"
        ],
        "Entertainment & Recreation": [
            "netflix", "spotify", "hulu", "disney plus", "hbo max",
            "youtube premium", "twitch", "apple music", "tidal",
            "peacock", "paramount", "espn plus", "discovery plus", "masterclass",
            "movie theater", "amc theaters", "regal cinemas", "concert ticket",
            "eventbrite", "stubhub", "ticketmaster", "steam games", "playstation",
            "netflix subscription", "spotify premium", "hulu monthly",
            "disney plus subscription", "youtube premium subscription"
        ],
        "Shopping & Retail": [
            "walmart", "target", "amazon purchase", "ebay", "best buy",
            "costco", "sams club", "home depot", "lowes", "ikea",
            "zara", "hm clothing", "gap store", "old navy", "uniqlo",
            "nordstrom", "macys", "kohls", "tjmaxx", "marshalls",
            "etsy purchase", "shein", "zappos shoes", "nike store",
            "adidas", "foot locker", "dick sporting goods", "rei outdoor",
            "amazon prime", "target store", "best buy electronics"
        ],
        "Healthcare & Medical": [
            "cvs pharmacy", "walgreens prescription", "rite aid", "hospital bill",
            "urgent care", "dental clinic", "eye doctor", "dermatologist",
            "physical therapy", "chiropractor", "labcorp", "quest diagnostics",
            "health insurance", "gym membership", "planet fitness", "equinox",
            "yoga studio", "headspace", "calm app", "peloton", "anytime fitness",
            "orange theory", "crunch fitness", "walgreens", "cvs"
        ],
        "Utilities & Services": [
            "rent payment", "mortgage payment", "electric bill", "gas utility",
            "water utility", "internet service", "comcast", "verizon fios",
            "att internet", "spectrum cable", "property tax", "renters insurance",
            "home insurance", "maintenance repair", "plumber service",
            "electrician", "pest control", "lawn service", "storage unit",
            "adobe creative", "microsoft 365", "dropbox", "slack", "zoom",
            "apple icloud", "linkedin premium", "audible", "kindle",
            "verizon bill", "comcast internet", "electric bill payment",
            "monthly rent", "rent", "utilities"
        ],
        "Financial Services": [
            "venmo transfer", "zelle payment", "paypal transfer", "cashapp send",
            "bank transfer", "wire transfer", "atm withdrawal", "check deposit",
            "western union", "money gram", "revolut transfer", "wise transfer",
            "square cash", "google pay", "apple pay transfer", "coinbase",
            "robinhood", "fidelity", "schwab", "vanguard", "etrade",
            "unknown wire transfer", "international wire", "zelle", "venmo"
        ],
        "Income": [
            "paycheck direct deposit", "freelance payment", "tax refund",
            "salary deposit", "bonus payment", "commission deposit",
            "dividend payment", "interest earned", "cashback reward",
            "refund credit", "reimbursement deposit", "rental income",
            "consulting fee", "side hustle payment", "venmo received",
            "direct deposit", "payroll deposit"
        ],
        "Charity & Donations": [
            "red cross donation", "gofundme", "patreon donation", "church offering",
            "salvation army", "goodwill donation", "united way", "habitat humanity",
            "doctors without borders", "wikipedia donation", "aclu donation"
        ],
        "Government & Legal": [
            "irs payment", "state tax payment", "dmv fee", "passport fee",
            "court fee", "parking ticket", "speeding ticket", "property tax",
            "municipal fine", "government fee", "social security", "medicare"
        ],
    }

    rows = []
    for cat, keywords in templates.items():
        for kw in keywords:
            for variant in [kw, f"{kw} purchase", f"online {kw}", f"{kw} - debit",
                            f"pmt {kw}", f"{kw} charge", f"{kw} transaction"]:
                rows.append({"description": variant, "category": cat})

    df = pd.DataFrame(rows)
    print(f"Fallback loaded: {len(df)} rows across {df['category'].nunique()} categories")
    return df
