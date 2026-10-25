from tool.size import money_short, size_bucket, size_label


def test_buckets():
    assert size_bucket(4.9e12) == "Mega" and size_bucket(150e9) == "Large" and size_bucket(13e9) == "Large"
    assert size_bucket(5e9) == "Mid" and size_bucket(1e9) == "Small" and size_bucket(50e6) == "Micro"
    assert size_bucket(None) is None and size_bucket(0) is None


def test_labels():
    assert size_label(4954724110000.0) == "Mega · $4.95T"
    assert size_label(150541777709.0) == "Large · $151B"
    assert money_short(18.96e9) == "$19.0B"
