import urllib.request


def test_headless_server_serves_the_app(streamlit_server):
    """
    Smoke test against the real server (started the way the hosting
    service runs the app): the health endpoint answers 'ok' and the
    start page is served. Ticket 0002 added this check, because the
    Streamlit update also replaced the server technology.
    """
    with urllib.request.urlopen(f'{streamlit_server}/_stcore/health',
                                timeout=10) as response:
        assert response.read() == b'ok'
    with urllib.request.urlopen(f'{streamlit_server}/', timeout=10) as response:
        assert response.status == 200
