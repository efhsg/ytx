"""Tests for --push config validation (graceful warning instead of hard exit)."""

import sys

import pytest

from ytx import load_config, main


class TestPushWithoutConfig:
    """When --push is used without PromptManager config, ytx should warn and skip push."""

    def test_missing_url_warns_and_continues(self, monkeypatch, capsys):
        """--push without promptmanager_url prints warning, does not exit."""
        monkeypatch.delenv('YTX_PROMPTMANAGER_URL', raising=False)
        monkeypatch.delenv('YTX_PROMPTMANAGER_TOKEN', raising=False)
        monkeypatch.setattr('ytx.os.path.exists', lambda p: False)

        # Stub transcript/metadata fetches to isolate config validation
        monkeypatch.setattr('ytx.fetch_video_metadata', lambda vid: {
            'title': 'Test', 'channel': 'Ch', 'channel_url': '',
            'description': '', 'tags': [], 'upload_date': '',
        })
        monkeypatch.setattr('ytx.get_transcript', lambda vid, lang='en': [])
        monkeypatch.setattr('ytx.transcript_to_text', lambda t: 'hello world')

        monkeypatch.setattr(sys, 'argv', ['ytx.py', 'dQw4w9WgXcQ', '--push'])
        main()  # Should NOT raise SystemExit

        stderr = capsys.readouterr().err
        assert 'WARNING' in stderr
        assert 'skipping push' in stderr

    def test_missing_token_warns_and_continues(self, monkeypatch, capsys):
        """--push with URL but no token prints warning, does not exit."""
        monkeypatch.setenv('YTX_PROMPTMANAGER_URL', 'http://localhost:8503')
        monkeypatch.delenv('YTX_PROMPTMANAGER_TOKEN', raising=False)
        monkeypatch.setattr('ytx.os.path.exists', lambda p: False)

        monkeypatch.setattr('ytx.fetch_video_metadata', lambda vid: {
            'title': 'Test', 'channel': 'Ch', 'channel_url': '',
            'description': '', 'tags': [], 'upload_date': '',
        })
        monkeypatch.setattr('ytx.get_transcript', lambda vid, lang='en': [])
        monkeypatch.setattr('ytx.transcript_to_text', lambda t: 'hello world')

        monkeypatch.setattr(sys, 'argv', ['ytx.py', 'dQw4w9WgXcQ', '--push'])
        main()

        stderr = capsys.readouterr().err
        assert 'WARNING' in stderr
        assert 'token' in stderr.lower()


class TestLoadConfig:
    """Config loading from env vars."""

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv('YTX_PROMPTMANAGER_URL', 'http://localhost:8503')
        monkeypatch.setenv('YTX_PROMPTMANAGER_TOKEN', 'test-token')
        monkeypatch.setattr('ytx.os.path.exists', lambda p: False)

        config = load_config()
        assert config['promptmanager_url'] == 'http://localhost:8503'
        assert config['promptmanager_token'] == 'test-token'

    def test_empty_config_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv('YTX_PROMPTMANAGER_URL', raising=False)
        monkeypatch.delenv('YTX_PROMPTMANAGER_TOKEN', raising=False)
        monkeypatch.setattr('ytx.os.path.exists', lambda p: False)

        config = load_config()
        assert config['promptmanager_url'] == ''
        assert config['promptmanager_token'] == ''
