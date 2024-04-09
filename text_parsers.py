from collections import defaultdict

def parse_boolean(text: str) -> bool | None:
    boolean_map = defaultdict(
        None,
        {
            "true": True,
            "false": False,
        },
    )

    clean_text = text.strip().split("\n")[0].lower()
    boolean = boolean_map.get(clean_text)
    return boolean


def parse_float(text: str):
    try:
        float_text = float(text.strip().split()[0])
    except Exception:
        return 0

    return float_text
