from src.scoring import normalize_text, precision_recall_f1, set_of_normalised


def test_normalize_text_basic():
    s = "  Hello, WORLD!! This (is) a test... "
    n = normalize_text(s)
    assert "hello world this is a test" == n


def test_precision_recall_f1_basic():
    preds = ["A", "B"]
    gold = ["A", "C"]
    prec, rec, f1 = precision_recall_f1(preds, gold)
    assert round(prec, 3) == 0.5
    assert round(rec, 3) == 0.5
    assert round(f1, 3) == 0.5


def test_empty_sets_return_ones():
    prec, rec, f1 = precision_recall_f1([], [])
    assert prec == 1.0 and rec == 1.0 and f1 == 1.0


def test_duplicates_handled():
    preds = ["Req", "Req", "Other"]
    gold = ["Req", "Other"]
    prec, rec, f1 = precision_recall_f1(preds, gold)
    assert prec == 1.0 and rec == 1.0 and f1 == 1.0
