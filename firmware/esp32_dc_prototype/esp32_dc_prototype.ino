/*
 * esp32_dc_prototype — timed, open-loop DC-motor control for the chess crane.
 *
 * PROTOTYPE firmware. Unlike esp32_chess.ino (which positions stepper motors by
 * counting steps), this drives ONE brushed DC motor through a 2-channel relay
 * module wired as an H-bridge, and controls motion purely by TIME: "run forward
 * for N milliseconds, then stop". There is no position feedback yet — the plan
 * is to hard-code square angles/durations and, if timing drifts too much, add a
 * Pi camera for visual correction later.

 * H-BRIDGE FROM 2 SPDT RELAYS
 * ---------------------------
 * Each motor terminal connects to the COM of one relay. Each relay's NC contact
 * ties to GND and its NO contact ties to the motor supply V+:
 *
 *        V+ ─── NO┐                 ┌NO ─── V+
 *                 relay A (CH1)      relay B (CH2)
 *        GND ── NC┘  COM─┐      ┌─COM └NC ── GND
 *                        │      │
 *                     term1 [ MOTOR ] term2
 *
 *   A energized, B released  -> term1=V+, term2=GND  -> FORWARD
 *   A released,  B energized  -> term1=GND, term2=V+  -> REVERSE
 *   both the same             -> both terminals equal -> STOP (no current)
 *
 * Because each relay only ever picks V+ *or* GND for its own terminal, the
 * supply can never be shorted through the bridge — there is no shoot-through to
 * worry about. We still pass through STOP with a short dead-time when reversing,
 * so the (slow, bouncy) relay contacts settle and the motor isn't slammed.
 *
 * WIRING NOTES
 *  - Power the motor from its OWN supply sized for the motor; do NOT run motor
 *    current through the ESP32. Tie the motor-supply GND to the ESP32 GND.
 *  - Put a snubber (e.g. 0.1uF + 100R, or a bidirectional TVS) across the motor
 *    terminals to tame the inductive arc when relays open. A single flyback
 *    diode won't do here because the motor reverses polarity.
 *  - Cheap relay boards are usually ACTIVE-LOW (IN pin LOW = relay energized).
 *    Set RELAY_ACTIVE_LOW below to match yours.
 *
 * LINE PROTOCOL (115200 baud, one command per line, '\n'-terminated):
 *   PING            -> OK PONG
 *   FWD [ms]        -> drive forward; with ms, auto-stops after ms. -> OK
 *   REV [ms]        -> drive reverse; with ms, auto-stops after ms. -> OK
 *   STOP            -> stop the motor now. -> OK
 *   ESTOP           -> stop + latch off until CLEAR. -> OK ESTOP
 *   CLEAR           -> release an ESTOP latch. -> OK
 *   STATUS          -> OK DIR<F|R|S> RUN<0|1> ESTOP<0|1>
 * A command with no ms runs until the next STOP. Timed runs are non-blocking:
 * the reply is immediate and loop() stops the motor when the timer expires, so
 * STOP/ESTOP can still interrupt a run in progress. Lines starting with '#' are
 * debug; 'EVT ...' lines are async notifications (e.g. a timed run finishing).
 */

// ======================= EDIT: pins & relay polarity =========================
#define RELAY_A_PIN      18      // CH1 -> motor terminal 1 (COM of relay A)
#define RELAY_B_PIN      19      // CH2 -> motor terminal 2 (COM of relay B)
const bool RELAY_ACTIVE_LOW = true;   // most hobby relay boards: LOW = energized

// ==================== EDIT: behaviour ========================================
const unsigned long RELAY_SETTLE_MS = 30;     // dead-time when changing direction
const unsigned long MAX_RUN_MS      = 10000;  // safety cap on a single timed run
// =============================================================================

enum Dir { STOPPED, FORWARD, REVERSE };

Dir           g_dir       = STOPPED;
bool          g_estopped  = false;
bool          g_running   = false;   // motor currently energized
bool          g_timed     = false;   // running against a deadline
unsigned long g_runUntil  = 0;       // millis() deadline for a timed run

char    g_line[64];
uint8_t g_len = 0;

// ----------------------------- relay layer -----------------------------------
// Write a logical "energized?" to a relay pin, honouring board polarity.
void relayWrite(int pin, bool energized) {
  digitalWrite(pin, (energized != RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

// Set both relays for a direction. Pass through STOP with a dead-time whenever
// the direction actually changes, so the contacts settle first.
void setDir(Dir dir) {
  if (dir == g_dir) return;

  // de-energize both (STOP) before re-energizing in the new configuration
  relayWrite(RELAY_A_PIN, false);
  relayWrite(RELAY_B_PIN, false);
  if (dir != STOPPED) delay(RELAY_SETTLE_MS);

  if (dir == FORWARD) {
    relayWrite(RELAY_A_PIN, true);
    relayWrite(RELAY_B_PIN, false);
  } else if (dir == REVERSE) {
    relayWrite(RELAY_A_PIN, false);
    relayWrite(RELAY_B_PIN, true);
  }
  g_dir = dir;
  g_running = (dir != STOPPED);
}

void stopMotor() {
  setDir(STOPPED);
  g_timed = false;
  g_runUntil = 0;
}

// ----------------------------- replies ----------------------------------------
void replyOK(const char* extra = nullptr) {
  if (extra && *extra) { Serial.print("OK "); Serial.println(extra); }
  else                 { Serial.println("OK"); }
}
void replyErr(const char* msg) { Serial.print("ERR "); Serial.println(msg); }

// ----------------------------- motion helpers ---------------------------------
// Start a run in `dir`. arg == the raw text after the command (may hold ms), or
// nullptr / empty for a continuous run until STOP.
void startRun(Dir dir, const char* arg) {
  long ms = 0;
  if (arg && *arg) ms = atol(arg);
  if (ms < 0) ms = 0;
  if ((unsigned long)ms > MAX_RUN_MS) ms = MAX_RUN_MS;

  setDir(dir);
  if (ms > 0) {
    g_timed = true;
    g_runUntil = millis() + (unsigned long)ms;
  } else {
    g_timed = false;   // continuous until STOP
    g_runUntil = 0;
  }
  replyOK();
}

void doStatus() {
  char buf[48];
  char d = (g_dir == FORWARD) ? 'F' : (g_dir == REVERSE) ? 'R' : 'S';
  snprintf(buf, sizeof(buf), "DIR%c RUN%d ESTOP%d",
           d, g_running ? 1 : 0, g_estopped ? 1 : 0);
  replyOK(buf);
}

// --------------------------- command dispatch --------------------------------
void handleLine(char* line) {
  char* cmd = strtok(line, " ");
  if (!cmd) return;
  char* arg = strtok(nullptr, " ");   // first argument, if any

  if (!strcmp(cmd, "PING")) {
    replyOK("PONG");
  } else if (!strcmp(cmd, "STATUS")) {
    doStatus();
  } else if (!strcmp(cmd, "STOP")) {
    stopMotor();
    replyOK();
  } else if (!strcmp(cmd, "ESTOP")) {
    stopMotor();
    g_estopped = true;
    replyOK("ESTOP");
  } else if (!strcmp(cmd, "CLEAR")) {
    g_estopped = false;
    replyOK();
  } else if (g_estopped) {
    replyErr("estopped; send CLEAR first");
  } else if (!strcmp(cmd, "FWD")) {
    startRun(FORWARD, arg);
  } else if (!strcmp(cmd, "REV")) {
    startRun(REVERSE, arg);
  } else {
    replyErr("unknown command");
  }
}

// ------------------------------- setup/loop ----------------------------------
void setup() {
  // Drive relays to de-energized BEFORE enabling them as outputs, so we don't
  // click the motor on at boot. (Set the level first, then pinMode.)
  relayWrite(RELAY_A_PIN, false);
  relayWrite(RELAY_B_PIN, false);
  pinMode(RELAY_A_PIN, OUTPUT);
  pinMode(RELAY_B_PIN, OUTPUT);
  relayWrite(RELAY_A_PIN, false);
  relayWrite(RELAY_B_PIN, false);

  Serial.begin(115200);
  Serial.println("# esp32_dc_prototype ready");
}

void loop() {
  // Stop a timed run when its deadline passes.
  if (g_timed && (long)(millis() - g_runUntil) >= 0) {
    stopMotor();
    Serial.println("EVT DONE");
  }

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (g_len > 0) {
        g_line[g_len] = '\0';
        handleLine(g_line);
        g_len = 0;
      }
    } else if (g_len < sizeof(g_line) - 1) {
      g_line[g_len++] = c;
    }
  }
}
