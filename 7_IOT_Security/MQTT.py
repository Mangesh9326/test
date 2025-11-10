import paho.mqtt.client as mqtt
import ssl

def on_connect(client, userdata, flags, rc, properties):
    print(f"\nConnected with result code {rc}")
    client.subscribe("iot/device1")

def on_message(client, userdata, msg):
    print(f"Received: {msg.topic} {msg.payload.decode()}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)

client.connect("test.mosquitto.org", 8883, 60)

client.loop_start()

client.publish("iot/device1", "Secured Temperature Data: 25C")

import time
time.sleep(5)

client.loop_stop()
client.disconnect()