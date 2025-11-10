import socket
from datetime import datetime

# -------------------------------
# Vulnerability Database (Dummy)
# -------------------------------
vulnerabilities = {
    "Apache/2.2.8": "Outdated Apache version - vulnerable to CVE-2009-3555",
    "OpenSSH_4.7": "Old OpenSSH version - possible remote exploits",
    "Microsoft-IIS/6.0": "IIS 6.0 is deprecated - vulnerable to multiple attacks",
}

# -------------------------------
# Port Scanner
# -------------------------------
def scan_ports(target, ports):
    open_ports = []
    print(f"\n[*] Scanning {target} ...")
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((target, port))
            if result == 0:
                print(f"[+] Port {port} is OPEN")
                open_ports.append(port)
            sock.close()
        except Exception:
            # ignore errors for individual ports
            pass
    return open_ports

# -------------------------------
# Banner Grabbing
# -------------------------------
def grab_banner(ip, port):
    sock = None
    try:
        sock = socket.socket()
        sock.settimeout(1.0)
        sock.connect((ip, port))
        # try to receive banner (some services send banner, some don't)
        raw = sock.recv(1024)
        try:
            banner = raw.decode(errors="replace").strip()
        except Exception:
            banner = None
        return banner
    except Exception:
        return None
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass

# -------------------------------
# Vulnerability Check
# -------------------------------
def check_vulnerabilities(banner):
    if not banner:
        return "No banner to check."
    for vuln in vulnerabilities:
        if vuln in banner:
            return vulnerabilities[vuln]
    return "No known vulnerability in database."

# -------------------------------
# Main Penetration Testing Script
# -------------------------------
if __name__ == "__main__":
    target = input("Enter Target IP (e.g., 127.0.0.1): ").strip()
    # common ports
    ports_to_scan = [21, 22, 23, 25, 80, 110, 143, 443, 3389]
    start_time = datetime.now()

    open_ports = scan_ports(target, ports_to_scan)

    print("\n[*] Banner Grabbing and Vulnerability Check:")
    for port in open_ports:
        banner = grab_banner(target, port)
        if banner:
            print(f"\n[Port {port}] Service Banner: {banner}")
            print(" -> " + check_vulnerabilities(banner))
        else:
            print(f"\n[Port {port}] No banner retrieved.")

    end_time = datetime.now()
    print("\n[*] Scan completed in:", end_time - start_time)
