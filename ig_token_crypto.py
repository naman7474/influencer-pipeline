"""Decrypt Instagram credential blobs encrypted by the web layer.

Mirror of web/src/lib/instagram/token-encryption.ts. The web side encrypts
with AES-256-GCM after deriving a 32-byte key via scrypt(password=IG_TOKEN_ENC_KEY,
salt=b'ig-token-salt-v1', n=16384, r=8, p=1, dklen=32). Stored format:

    "ig-token-v1:" + base64( iv (12 bytes) | tag (16 bytes) | ciphertext )

For the cookie-mode path (personal_token_kind='apify_actor'), the plaintext
is a JSON array:

    [{"name":"sessionid","value":"..."}, {"name":"csrftoken",...}, ...]

This module provides ``decrypt_token(stored)`` returning the plaintext
string, and ``decode_ig_cookies(stored)`` returning the parsed JSON array.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_INFO_PREFIX = "ig-token-v1:"
_SALT = b"ig-token-salt-v1"
_KEY_LEN = 32
_IV_LEN = 12
_TAG_LEN = 16
# Node scrypt defaults — must match scryptSync(secret, SALT, 32) in the TS code.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1


def _derive_key() -> bytes:
    secret = os.environ.get("IG_TOKEN_ENC_KEY")
    if not secret or len(secret) < 16:
        raise RuntimeError(
            "IG_TOKEN_ENC_KEY env var must be set to a strong secret (32+ chars recommended)"
        )
    return hashlib.scrypt(
        password=secret.encode("utf-8"),
        salt=_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LEN,
    )


def decrypt_token(stored: str) -> str:
    """Decrypt the AES-256-GCM blob produced by the web layer.

    Tokens written before encryption was introduced (legacy rows from
    migration 019) lack the prefix and are returned as-is for parity
    with the TS ``decryptToken`` fallback.
    """
    if not stored.startswith(_KEY_INFO_PREFIX):
        return stored

    raw = base64.b64decode(stored[len(_KEY_INFO_PREFIX):])
    iv = raw[:_IV_LEN]
    tag = raw[_IV_LEN:_IV_LEN + _TAG_LEN]
    ciphertext = raw[_IV_LEN + _TAG_LEN:]

    # AESGCM expects ciphertext + tag concatenated.
    aes = AESGCM(_derive_key())
    plaintext = aes.decrypt(iv, ciphertext + tag, associated_data=None)
    return plaintext.decode("utf-8")


def decode_ig_cookies(stored: str) -> list[dict[str, Any]]:
    """Decrypt and JSON-parse the cookies blob written by the apify_actor flow.

    The stored value (brand_instagram_accounts.access_token) contains a JSON
    array — one element per cookie — produced by the
    "Export Cookie JSON file for Puppeteer" extension. Returns the array.
    """
    plain = decrypt_token(stored)
    parsed = json.loads(plain)
    if not isinstance(parsed, list):
        raise ValueError("decoded cookies blob is not a JSON array")
    return parsed
