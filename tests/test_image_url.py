import pytest

from inkling_engine.server import check_image_url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.png", "/etc/passwd", "http://127.0.0.1/a.png",
                                 "http://localhost:8000/a.png", "http://10.0.0.1/a.png", "http://169.254.169.254/latest",
                                 "http://[::1]/a.png", "http://", "http://[64:ff9b::a9fe:a9fe]/a.png",
                                 "http://[2002:7f00:1::]/a.png", "http://224.0.0.1/a.png"])
def test_rejected(url):
    with pytest.raises(ValueError):
        check_image_url(url)


def test_public_address_allowed():
    check_image_url("https://1.1.1.1/a.png")


def test_private_allowed_by_env(monkeypatch):
    monkeypatch.setenv("INKLING_LOCAL_IMAGES", "1")
    check_image_url("http://127.0.0.1/a.png")
