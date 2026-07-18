import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from backtest import _download_with_retries


class DownloadRetryTests(unittest.TestCase):
    def test_retries_transient_http_error_and_succeeds(self):
        attempts = {"count": 0}

        def fake_download():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise HTTPError(
                    url="https://example.com",
                    code=503,
                    msg="Service Unavailable",
                    hdrs=None,
                    fp=None,
                )
            return "ok"

        with patch("backtest.time.sleep", return_value=None):
            result = _download_with_retries(fake_download, "TEST", max_retries=3, delay_seconds=0)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 3)


if __name__ == "__main__":
    unittest.main()
