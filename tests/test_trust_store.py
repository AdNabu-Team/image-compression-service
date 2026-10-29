"""Tests for security.trust_store, and a guard that both images run it."""

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from security import trust_store
from security.trust_store import deselect_lines, filter_bundle, is_distrusted

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILES = ("Dockerfile", "Dockerfile.jxl")


def root_certificate(organisation: str, common_name: str) -> str:
    """A self-signed CA certificate for the organisation, as PEM."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organisation),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM).decode().strip()


@pytest.fixture(scope="module")
def trusted():
    return root_certificate("Google Trust Services LLC", "GTS Root R1")


@pytest.fixture(scope="module")
def distrusted():
    return root_certificate("eMudhra Technologies Limited", "emSign ECC Root CA - G3")


def certifi_entry(label: str, pem: str) -> str:
    """One certifi-style entry: its comment lines, then the certificate."""
    return f'\n# Issuer: CN={label}\n# Label: "{label}"\n{pem}\n'


def test_filter_removes_distrusted_cas_and_their_comments(trusted, distrusted):
    bundle = certifi_entry("emSign ECC Root CA - G3", distrusted) + certifi_entry("GTS", trusted)

    cleaned, removed = filter_bundle(bundle)

    assert removed == 1
    assert certifi_entry("GTS", trusted).strip() in cleaned
    assert "emSign" not in cleaned


def test_a_bundle_without_distrusted_cas_is_unchanged(trusted):
    bundle = certifi_entry("GTS", trusted)
    assert filter_bundle(bundle) == (bundle, 0)


def test_judges_the_certificate_not_its_comment(trusted, distrusted):
    assert is_distrusted(distrusted)
    assert not is_distrusted(trusted)
    assert not is_distrusted(
        "-----BEGIN CERTIFICATE-----\nbm90IGEgY2VydA==\n-----END CERTIFICATE-----"
    )


def test_only_distrusted_config_entries_are_deselected():
    lines = ["# comment", "mozilla/GTS_Root_R1.crt", "mozilla/emSign_Root_CA_-_G1.crt"]
    assert deselect_lines(lines, {"mozilla/emSign_Root_CA_-_G1.crt"}) == [
        "# comment",
        "mozilla/GTS_Root_R1.crt",
        "!mozilla/emSign_Root_CA_-_G1.crt",
    ]


@pytest.fixture
def stores(tmp_path, trusted, distrusted):
    """certifi and Debian stores under tmp_path, each holding one trusted and one distrusted CA."""
    certifi_path = tmp_path / "cacert.pem"
    certifi_path.write_text(certifi_entry("GTS", trusted) + certifi_entry("emSign", distrusted))
    source = tmp_path / "share" / "mozilla"
    source.mkdir(parents=True)
    (source / "GTS_Root_R1.crt").write_text(trusted)
    (source / "emSign_Root_CA_-_G1.crt").write_text(distrusted)
    config = tmp_path / "ca-certificates.conf"
    config.write_text("# comment\nmozilla/GTS_Root_R1.crt\nmozilla/emSign_Root_CA_-_G1.crt\n")
    os_bundle = tmp_path / "ca-certificates.crt"
    os_bundle.write_text(f"{trusted}\n")
    with (
        patch.object(trust_store.certifi, "where", return_value=str(certifi_path)),
        patch.object(trust_store, "OS_CA_CONFIG", str(config)),
        patch.object(trust_store, "OS_CA_SOURCE_DIR", str(tmp_path / "share")),
        patch.object(trust_store, "OS_CA_BUNDLE", str(os_bundle)),
        patch.object(trust_store.subprocess, "run") as run,
    ):
        yield {
            "certifi": certifi_path,
            "config": config,
            "os_bundle": os_bundle,
            "run": run,
            "distrusted": distrusted,
        }


def test_certifi_bundle_is_cleaned_in_place(stores, trusted):
    assert trust_store.clean_certifi_bundle() == 1
    assert trust_store.distrusted_in_bundle(str(stores["certifi"])) == 0
    assert trusted in stores["certifi"].read_text()


def test_os_store_deselects_distrusted_entries_and_rebuilds(stores):
    assert trust_store.clean_os_bundle() == 1
    assert stores["config"].read_text() == (
        "# comment\nmozilla/GTS_Root_R1.crt\n!mozilla/emSign_Root_CA_-_G1.crt\n"
    )
    stores["run"].assert_called_once_with(
        ["update-ca-certificates", "--fresh"], check=True, stdout=trust_store.subprocess.DEVNULL
    )


def test_main_passes_when_both_bundles_are_clean(stores):
    trust_store.main()
    assert trust_store.distrusted_in_bundle(str(stores["certifi"])) == 0


def test_main_fails_the_build_while_a_bundle_still_trusts_one(stores, trusted):
    # update-ca-certificates is mocked, so this OS bundle keeps the distrusted CA
    stores["os_bundle"].write_text(f"{trusted}\n{stores['distrusted']}\n")
    with pytest.raises(SystemExit):
        trust_store.main()


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_every_image_runs_the_step(dockerfile):
    """An edited Dockerfile must keep the step, or its image trusts the compromised CAs again."""
    assert "RUN python -m security.trust_store" in (REPO_ROOT / dockerfile).read_text()


def test_no_dockerfile_is_left_out():
    assert {path.name for path in REPO_ROOT.glob("Dockerfile*")} == set(DOCKERFILES)
