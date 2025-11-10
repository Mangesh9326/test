# toy_collision_demo.py
# Educational demo: two different "certificate" byte-strings that produce the same value

def toy_hash(b: bytes) -> int:
    """
    Very weak toy hash: sum of bytes modulo 256.
    This is intentionally insecure and collision-prone — perfect for demonstration.
    """
    return sum(b) % 256

def make_base_certs():
    # Two different PEM-like certificate strings (plain text). Not real certs.
    cert1_txt = """-----BEGIN CERTIFICATE-----
Subject: CN=Alice Example
Issuer: Test CA
PublicKey: AAAAB3NzaC1yc2EAAAADAQABAAABAQCu
-----END CERTIFICATE-----
"""
    cert2_txt = """-----BEGIN CERTIFICATE-----
Subject: CN=Bob Example
Issuer: Test CA
PublicKey: AAAAB3NzaC1yc2EAAAADAQABAAABAQCu
-----END CERTIFICATE-----
"""
    return cert1_txt.encode('utf-8'), cert2_txt.encode('utf-8')

def pad_to_match(hash_target: int, current_bytes: bytes) -> bytes:
    """
    Append a small padding (bytes) to `current_bytes` so that
    toy_hash(current_bytes + padding) == hash_target (mod 256).
    Returns the padding bytes (may include non-printable bytes).
    This routine constructs padding using up to two bytes (0..255).
    """
    cur = sum(current_bytes) % 256
    needed = (hash_target - cur) % 256  # amount we must add (0..255)

    # If needed is 0, no padding required
    if needed == 0:
        return b''

    # Try single-byte padding if possible
    if 0 <= needed <= 255:
        return bytes([needed])

    # Fallback: represent needed as two bytes (shouldn't be needed here)
    hi = needed // 256
    lo = needed % 256
    return bytes([hi, lo])

def main():
    cert1, cert2 = make_base_certs()
    h1 = toy_hash(cert1)
    h2 = toy_hash(cert2)
    print("Initial toy hashes:")
    print(" cert1 hash:", h1)
    print(" cert2 hash:", h2)
    print()

    # We will pad cert2 so its toy_hash matches cert1
    padding = pad_to_match(h1, cert2)
    cert2_padded = cert2 + b"\n#PAD:" + padding  # append a comment plus padding bytes

    h2_after = toy_hash(cert2_padded)
    print("After padding cert2:")
    print(" cert1 length:", len(cert1), "hash:", toy_hash(cert1))
    print(" cert2_padded length:", len(cert2_padded), "hash:", h2_after)
    print()

    print("Are bytes identical?:", cert1 == cert2_padded)
    print("Do they collide under toy_hash?:", toy_hash(cert1) == toy_hash(cert2_padded))
    print()

    # For clarity, show short hex of last bytes of each
    def show_tail(b):
        tail = b[-40:]
        return tail.hex()

    print("tail(cert1)   (hex):", show_tail(cert1))
    print("tail(cert2_p)  (hex):", show_tail(cert2_padded))

    # If you want to write them to files for inspection:
    with open("cert1_demo.pem", "wb") as f:
        f.write(cert1)
    with open("cert2_demo_padded.pem", "wb") as f:
        f.write(cert2_padded)

    print("\nWrote cert1_demo.pem and cert2_demo_padded.pem to disk (different files).")
    print("You can show students: files differ, but toy_hash(...) is the same.")

if __name__ == "__main__":
    main()
