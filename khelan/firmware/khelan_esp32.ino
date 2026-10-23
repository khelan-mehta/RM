/*
 * Khelan — reference ESP32 firmware
 * ---------------------------------
 * Connects an ESP32 to the Khelan backend over WebSocket and controls one
 * output (relay / LED on RELAY_PIN). It mirrors a virtual device: it receives
 * {cmd:"set", state:{on, brightness, ...}} and drives the pin, and reports its
 * state back with {type:"state", state:{...}}. It can also push a sensor or
 * sound event.
 *
 * Arduino IDE setup:
 *   - Boards Manager: "esp32" by Espressif
 *   - Libraries: "WebSockets" by Markus Sattler (links2004/arduinoWebSockets),
 *                "ArduinoJson" by Benoit Blanchon
 *   - Select your ESP32 board + port, then Upload.
 *
 * Provision a device in the Khelan dashboard: + Device → Kind = "ESP32".
 * Copy the shown HOST, PORT and TOKEN into the CONFIG section below.
 * For the HTTPS/--remote server use WebSocketsClientSecure (beginSSL) instead.
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// ---------------- CONFIG ----------------
const char* WIFI_SSID = "your-wifi";
const char* WIFI_PASS = "your-wifi-password";
const char* HOST      = "192.168.1.6";      // Khelan server LAN IP
const uint16_t PORT   = 8000;               // 8000 local, 8443 for --remote (use SSL)
const char* TOKEN     = "paste-device-token-here";

const int RELAY_PIN = 2;                    // onboard LED on many ESP32 dev boards
// ----------------------------------------

WebSocketsClient ws;
bool deviceOn = false;
int brightness = 100;

void sendState() {
  StaticJsonDocument<128> doc;
  doc["type"] = "state";
  JsonObject st = doc.createNestedObject("state");
  st["on"] = deviceOn;
  st["brightness"] = brightness;
  String out; serializeJson(doc, out);
  ws.sendTXT(out);
}

void applyState(JsonObject st) {
  if (st.containsKey("on")) deviceOn = st["on"];
  if (st.containsKey("brightness")) brightness = st["brightness"];
  // Simple on/off relay. For PWM dimming use ledcWrite() with brightness.
  digitalWrite(RELAY_PIN, deviceOn ? HIGH : LOW);
  Serial.printf("Applied: on=%d brightness=%d\n", deviceOn, brightness);
}

void onEvent(WStype_t type, uint8_t* payload, size_t len) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("Khelan: connected");
      sendState();
      break;
    case WStype_DISCONNECTED:
      Serial.println("Khelan: disconnected");
      break;
    case WStype_TEXT: {
      StaticJsonDocument<256> doc;
      if (deserializeJson(doc, payload, len)) return;
      const char* cmd = doc["cmd"];
      if (cmd && strcmp(cmd, "set") == 0 && doc["state"].is<JsonObject>()) {
        applyState(doc["state"].as<JsonObject>());
        sendState();   // confirm actual state back to the server
      }
      break;
    }
    default: break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());

  String path = String("/esp32?token=") + TOKEN;
  ws.begin(HOST, PORT, path);           // ws.beginSSL(HOST, 8443, path) for --remote
  ws.onEvent(onEvent);
  ws.setReconnectInterval(3000);
}

void loop() {
  ws.loop();

  // Example: report a sensor/sound event (wire your detector here).
  // StaticJsonDocument<96> ev; ev["type"]="sound"; ev["sound"]="glass"; ev["confidence"]=0.95;
  // String s; serializeJson(ev, s); ws.sendTXT(s);
}
