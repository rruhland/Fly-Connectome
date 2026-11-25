from scripts.experimental.tt51_learned_commitment import Book


def test_duration_adapts_only_after_actual_outcome_updates():
    book = Book()
    assert book.choose(.2)[:3] == (0, 0, "cold-stock")
    book.observe(0, 0, True)
    assert book.choose(.2)[:3] == (0, 1, "bounded-hold-probe")
    book.observe(0, 1, False)
    assert book.choose(.2)[:3] == (0, 0, "outcome-weighted")
    book.observe(0, 0, False)
    for _ in range(6):
        book.observe(0, 1, True)
    assert book.choose(.2)[:3] == (0, 1, "outcome-weighted")
    assert Book(book.serializable()).choose(.2) == book.choose(.2)
