"""Remove distrusted certificate authorities from the image's trust stores.

Both Dockerfiles run it after copying the application: `python -m security.trust_store`.

eMudhra's emSign CAs were compromised (CVE-2026-111943, 20 Oct 2026): their intermediates'
private keys can be recovered from the public certificates. Browsers block them, but server-side
code trusts whatever its bundle holds, so they are removed from both bundles outbound HTTPS uses:

- Python's certifi bundle (httpx URL fetches, google-cloud-storage), edited in place;
- Debian's OS bundle (OpenSSL defaults), by deselecting them in /etc/ca-certificates.conf.

The build fails if any distrusted certificate is still in either bundle. Same as
adnabu-api's utils/trust_store.py.
"""

import os
import re
import subprocess
import sys
import warnings
from typing import Iterable

import certifi
from cryptography import x509
from cryptography.utils import CryptographyDeprecationWarning
from cryptography.x509.oid import NameOID

# Matched against each certificate's subject organisation and common name
DISTRUSTED_CA_PATTERN = re.compile(r"emsign|emudhra", re.IGNORECASE)
OS_CA_CONFIG = "/etc/ca-certificates.conf"
OS_CA_SOURCE_DIR = "/usr/share/ca-certificates"
OS_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
PEM_BLOCK = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", re.DOTALL)


def is_distrusted(pem: str) -> bool:
    """Whether the PEM certificate's subject names a distrusted CA."""
    # Some bundled roots have a non-positive serial number, which cryptography warns about and
    # will one day refuse; a certificate it can't parse is judged by nothing and kept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CryptographyDeprecationWarning)
        try:
            subject = x509.load_pem_x509_certificate(pem.encode()).subject
        except ValueError:
            return False
    names = [
        attribute.value
        for oid in (NameOID.ORGANIZATION_NAME, NameOID.COMMON_NAME)
        for attribute in subject.get_attributes_for_oid(oid)
    ]
    return any(DISTRUSTED_CA_PATTERN.search(str(name)) for name in names)


def filter_bundle(text: str) -> tuple[str, int]:
    """The bundle without its distrusted certificates and their comment lines, and how many."""
    kept, removed, position = [], 0, 0
    for match in PEM_BLOCK.finditer(text):
        preamble = text[position : match.start()]
        if is_distrusted(match.group(0)):
            removed += 1
            # certifi puts '# Issuer/Subject/Label' lines before each certificate: drop them too
            kept.append(re.sub(r"(?m)^#.*\n", "", preamble))
        else:
            kept.append(preamble + match.group(0))
        position = match.end()
    kept.append(text[position:])
    return "".join(kept), removed


def distrusted_in_bundle(path: str) -> int:
    """How many distrusted certificates the bundle at path still holds."""
    with open(path, encoding="ascii") as bundle:
        return sum(is_distrusted(block) for block in PEM_BLOCK.findall(bundle.read()))


def clean_certifi_bundle() -> int:
    """Remove the distrusted certificates from certifi's bundle in place; returns how many."""
    path = certifi.where()
    with open(path, encoding="ascii") as bundle:
        cleaned, removed = filter_bundle(bundle.read())
    with open(path, "w", encoding="ascii") as bundle:
        bundle.write(cleaned)
    return removed


def deselect_lines(config_lines: Iterable[str], distrusted_paths: set[str]) -> list[str]:
    """ca-certificates.conf lines with each distrusted entry deselected with a '!' prefix."""
    return [f"!{line}" if line.strip() in distrusted_paths else line for line in config_lines]


def clean_os_bundle() -> int:
    """Deselect the distrusted certificates in Debian's store and rebuild the OS bundle."""
    distrusted = set()
    for root, _dirs, files in os.walk(OS_CA_SOURCE_DIR):
        for name in files:
            if name.endswith(".crt"):
                path = os.path.join(root, name)
                with open(path, encoding="ascii") as certificate:
                    if is_distrusted(certificate.read()):
                        distrusted.add(os.path.relpath(path, OS_CA_SOURCE_DIR))
    with open(OS_CA_CONFIG, encoding="utf-8") as config:
        lines = config.read().splitlines()
    with open(OS_CA_CONFIG, "w", encoding="utf-8") as config:
        config.write("\n".join(deselect_lines(lines, distrusted)) + "\n")
    subprocess.run(["update-ca-certificates", "--fresh"], check=True, stdout=subprocess.DEVNULL)
    return len(distrusted)


def main() -> None:
    """Clean both bundles, then fail if either still trusts a distrusted CA."""
    certifi_removed, os_removed = clean_certifi_bundle(), clean_os_bundle()
    print(f"[trust_store] removed {certifi_removed} from certifi, {os_removed} from the OS store")
    remaining = {path: distrusted_in_bundle(path) for path in (certifi.where(), OS_CA_BUNDLE)}
    if any(remaining.values()):
        sys.exit(f"[trust_store] distrusted CAs still trusted: {remaining}")


if __name__ == "__main__":
    main()
