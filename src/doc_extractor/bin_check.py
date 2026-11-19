"""Контрольная цифра БИН/ИИН (12 цифр)."""

_W1 = range(1, 12)
_W2 = (3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2)


def bin_checksum_ok(bin12: str) -> bool:
    if len(bin12) != 12 or not bin12.isascii() or not bin12.isdigit():
        return False
    d = [int(c) for c in bin12]
    s = sum(w * x for w, x in zip(_W1, d[:11])) % 11
    if s == 10:
        s = sum(w * x for w, x in zip(_W2, d[:11])) % 11
        if s == 10:
            return False
    return s == d[11]
