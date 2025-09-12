import re, unicodedata

_RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
_RE_PHONE = re.compile(r"\b(?:\+?\d{1,3})?\s?(?:\(?\d{2}\)?)?\s?\d{4,5}-?\d{4,5}\b")
_RE_CPF   = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_RE_CNPJ  = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
_RE_UUID  = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_RE_NUM   = re.compile(r"\b\d+([.,]\d+)?\b")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold().strip()
    s = _RE_EMAIL.sub("<EMAIL>", s)
    s = _RE_PHONE.sub("<PHONE>", s)
    s = _RE_CPF.sub("<CPF>", s)
    s = _RE_CNPJ.sub("<CNPJ>", s)
    s = _RE_UUID.sub("<UUID>", s)
    # valores sem "R$": manter só o número como placeholder
    s = s.replace("r$", "")  # caso apareça
    s = _RE_NUM.sub("<VAL>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s