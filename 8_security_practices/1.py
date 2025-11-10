def toy_hash(b: bytes) -> int:
   return sum(b) % 256

def make_base_certs():
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

   cur = sum(current_bytes) % 256
   needed = (hash_target - cur) % 256  
   if needed == 0:
       return b''

   if 0 <= needed <= 255:
       return bytes([needed])
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
  
   padding = pad_to_match(h1, cert2)
   cert2_padded = cert2 + b"\n#PAD:" + padding 
   h2_after = toy_hash(cert2_padded)
  
   print("After padding cert2:")
   print(" cert1 length:", len(cert1), "hash:", toy_hash(cert1))
   print(" cert2_padded length:", len(cert2_padded), "hash:", h2_after)
   print()
  
   print("Are bytes identical?:", cert1 == cert2_padded)
   print("Do they collide under toy_hash?:", toy_hash(cert1) == toy_hash(cert2_padded))
   print()
  
   def show_tail(b):
       tail = b[-40:]
       return tail.hex()
  
   print("tail(cert1) (hex):", show_tail(cert1))
   print("tail(cert2_p) (hex):", show_tail(cert2_padded))
  
   with open("cert1_demo.pem", "wb") as f:
       f.write(cert1)
   with open("cert2_demo_padded.pem", "wb") as f:
       f.write(cert2_padded)
  
   
if __name__ == "__main__":
   main()

