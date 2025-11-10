from scapy.all import sniff

def packet_callback(packet):
   if packet.haslayer("IP"):
       src = packet["IP"].src
       dst = packet["IP"].dst
       print(f"[IP Packet] {src} → {dst}")
       if packet.haslayer("Raw"):
           payload = packet["Raw"].load
           print("Payload:", payload)

print(" Sniffing IoT traffic... Press Ctrl+C to stop")
sniff(prn=packet_callback)
