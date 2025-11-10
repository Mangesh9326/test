import json
from datetime import datetime

blacklisted_ips = ["192.168.1.10", "10.0.0.5"]

with open("logs.json") as f:
    logs = json.load(f)

failed_attempts = {}

print("\n Intrusion Detection Log Analysis:\n")

for entry in logs:
    user = entry["user"]
    ip = entry["ip"]
    time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
    status = entry["status"]

    if ip in blacklisted_ips:
        print(f" ALERT: Access from blacklisted IP {ip} by user '{user}'")

    if not (8 <= time.hour <= 18):
        print(f" Off-hour login attempt by '{user}' at {time.strftime('%H:%M')}")

    if status == "failed":
        failed_attempts[user] = failed_attempts.get(user, 0) + 1
        if failed_attempts[user] >= 2:
            print(f" User '{user}' has {failed_attempts[user]} failed login attempts!")

print("\n Log scan completed.\n")