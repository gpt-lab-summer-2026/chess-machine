/*
 * esp32_dc_prototype — timed, open-loop actuator control for the chess crane.
 *
 * PROTOTYPE firmware. Drives a mix of actuators over a simple serial line
 * protocol, by TIME / STEP count (no position feedback yet):
 *
 *   - Up to N brushed DC motors, each via a 2-relay H-bridge (FWD/REV/STOP).
 *     A motor may instead have a SINGLE relay (pinB = -1): on/off, one direction.
 *   - One stepper motor via a ULN2003 driver (28BYJ-48-style), its own STEP cmd.
 *
 * ACTUATORS (edit the tables below to match your wiring):
 *   M1  base (single relay)   relay on pin 19            FWD = on, STOP = off
 *   M2  boom rail (horiz.)    relays on pins  4, 15      H-bridge FWD/REV
 *   M3  lift wire (vert.)     relays on pins 22, 23      H-bridge FWD/REV
 *   STEP  stepper (ULN2003)   IN1..IN4 = 5, 18, 21, 17   STEP <signed steps>
 *
 *   NOTE ON PINS: GPIO3 is the USB-serial RX pin — using it as an output would
 *   kill command reception, so the 4th stepper coil is on GPIO17, not 3. GPIO5
 *   and GPIO15 are strapping pins (usable as outputs after boot). GPIO22/23 have
 *   no boot pull-up: hold relay inputs HIGH with external ~10k pull-ups to VCC
 *   (active-low boards) or a motor may lurch on power-up before setup() runs.
 *
 * H-BRIDGE FROM 2 SPDT RELAYS (per DC motor)
 * ------------------------------------------
 * Each motor terminal -> COM of one relay; NC -> GND, NO -> motor supply V+:
 *   A energized, B released  -> term1=V+, term2=GND  -> FORWARD
 *   A released,  B energized  -> term1=GND, term2=V+  -> REVERSE
 *   both the same             -> both terminals equal -> STOP (no current)
 * Each relay only picks V+ *or* GND for its own terminal, so the supply can't be
 * shorted through the bridge. We pass through STOP with a dead-time on reversal.
 * A SINGLE-relay motor (pinB = -1) hard-wires term2; FWD energizes, STOP releases.
 *
 * WIRING NOTES
 *  - Power motors + stepper from their OWN supply, common ground with the ESP32.
 *    The 28BYJ-48 runs from the ULN2003 board's 5V; do not back-feed the ESP32.
 *  - Snubber (0.1uF + 100R or a bidirectional TVS) across each DC motor.
 *
 * LINE PROTOCOL (115200 baud, one command per line). <m> = DC motor index 1..N:
 *   PING              -> OK PONG
 *   FWD <m> [ms]      -> DC motor m forward; with ms, auto-stops after ms. -> OK
 *   REV <m> [ms]      -> DC motor m reverse (H-bridge motors only). -> OK
 *   STOP [m]          -> stop DC motor m, or ALL (motors + stepper) if omitted.
 *   STEP <steps> [ms] -> stepper: move `steps` (signed; + / - = direction),
 *                        `ms` = delay between steps (default below). Blocks. -> OK
 *   ESTOP             -> stop everything + latch off until CLEAR. -> OK ESTOP
 *   CLEAR             -> release an ESTOP latch. -> OK
 *   STATUS            -> OK M1:<F|R|S> ... STEP:<pos> ESTOP<0|1>
 * DC FWD/REV with no ms run until the next STOP, and are non-blocking. STEP runs
 * to completion (blocking), but DC timers are still serviced meanwhile. Lines
 * starting with '#' are debug; 'EVT ...' are async (e.g. `EVT DONE M2`).
 */

// ======================= EDIT: relay polarity ================================
const bool RELAY_ACTIVE_LOW = true;   // most hobby relay boards: LOW = energized

// ==================== EDIT: behaviour ========================================
const unsigned long RELAY_SETTLE_MS = 30;      // dead-time when changing direction
const unsigned long MAX_RUN_MS      = 10000;   // safety cap on a single timed run
const unsigned long STEP_DELAY_MS   = 2;       // default delay between stepper steps
const long          STEP_MAX        = 100000;  // safety cap on one STEP command
// =============================================================================

enum Dir { STOPPED, FORWARD, REVERSE };

struct Motor {
  const char*   name;
  int           pinA;      // relay A -> motor terminal 1
  int           pinB;      // relay B -> motor terminal 2, or -1 for a single relay
  Dir           dir;
  bool          timed;
  unsigned long runUntil;
};

// ======================= EDIT: DC motors & pins ==============================
Motor motors[] = {
  { "base relay",  19, -1, STOPPED, false, 0 },   // M1: single relay (on/off)
  { "boom rail",    4, 15, STOPPED, false, 0 },   // M2: H-bridge (horizontal)
  { "lift wire",   22, 23, STOPPED, false, 0 },   // M3: H-bridge (vertical)
};
const int NUM_MOTORS = sizeof(motors) / sizeof(motors[0]);

// ======================= EDIT: stepper (ULN2003) pins ========================
// Wire the ULN2003 IN1..IN4 to exactly these GPIOs. NONE of them may be GPIO1
// (TX0) or GPIO3 (RX0): those are the USB-serial link. Putting a coil on a UART
// pin leaves that coil undriven, so the motor only VIBRATES instead of turning.
#define STEP_IN1   5    // board IN1
#define STEP_IN2  21    // board IN2
#define STEP_IN3  18    // board IN3
#define STEP_IN4  17    // board IN4  (NOT TX0/RX0 — a free GPIO)
// If it still only vibrates with all four wired, the coil ORDER is wrong: swap
// the middle two (IN2 <-> IN3), or send the exact IN->GPIO map and re-derive it.
// =============================================================================

// 28BYJ-48 half-step sequence (IN1..IN4). Reverse = walk it backwards.
const uint8_t HALFSTEP[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1},
};

bool    g_estopped = false;
int     g_stepPhase = 0;   // current index into HALFSTEP
long    g_stepPos   = 0;   // net steps since boot (for STATUS)
char    g_line[64];
uint8_t g_len = 0;

// ----------------------------- relay layer -----------------------------------
void relayWrite(int pin, bool energized) {
  digitalWrite(pin, (energized != RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

// Set a motor's relay(s) for a direction, passing through STOP with a dead-time
// on any change. Single-relay motors (pinB < 0) only act on FORWARD.
void setDir(Motor& m, Dir dir) {
  if (dir == m.dir) return;

  relayWrite(m.pinA, false);
  if (m.pinB >= 0) relayWrite(m.pinB, false);
  if (dir != STOPPED) delay(RELAY_SETTLE_MS);

  if (m.pinB < 0) {
    if (dir == FORWARD) relayWrite(m.pinA, true);   // single relay: on/off only
  } else if (dir == FORWARD) {
    relayWrite(m.pinA, true);
    relayWrite(m.pinB, false);
  } else if (dir == REVERSE) {
    relayWrite(m.pinA, false);
    relayWrite(m.pinB, true);
  }
  m.dir = dir;
}

void stopMotor(Motor& m) {
  setDir(m, STOPPED);
  m.timed = false;
  m.runUntil = 0;
}

// ----------------------------- stepper layer ----------------------------------
void stepperWritePhase(int phase) {
  digitalWrite(STEP_IN1, HALFSTEP[phase][0]);
  digitalWrite(STEP_IN2, HALFSTEP[phase][1]);
  digitalWrite(STEP_IN3, HALFSTEP[phase][2]);
  digitalWrite(STEP_IN4, HALFSTEP[phase][3]);
}

void stepperRelease() {   // de-energize all coils (no holding torque, no heat)
  digitalWrite(STEP_IN1, LOW);
  digitalWrite(STEP_IN2, LOW);
  digitalWrite(STEP_IN3, LOW);
  digitalWrite(STEP_IN4, LOW);
}

// ----------------------------- replies ----------------------------------------
void replyOK(const char* extra = nullptr) {
  if (extra && *extra) { Serial.print("OK "); Serial.println(extra); }
  else                 { Serial.println("OK"); }
}
void replyErr(const char* msg) { Serial.print("ERR "); Serial.println(msg); }

// --------------------------- timers / motion ---------------------------------
// Stop each DC motor whose timed run has expired. Called from loop() AND from
// inside the (blocking) stepper move so DC timers stay honoured while stepping.
void serviceMotorTimers() {
  for (int i = 0; i < NUM_MOTORS; i++) {
    Motor& m = motors[i];
    if (m.timed && (long)(millis() - m.runUntil) >= 0) {
      stopMotor(m);
      Serial.print("EVT DONE M");
      Serial.println(i + 1);
    }
  }
}

void stopAll() {
  for (int i = 0; i < NUM_MOTORS; i++) stopMotor(motors[i]);
  stepperRelease();
}

Motor* motorArg(const char* tok) {
  if (!tok || !*tok) return nullptr;
  int idx = atoi(tok);
  if (idx < 1 || idx > NUM_MOTORS) return nullptr;
  return &motors[idx - 1];
}

void startRun(Motor& m, Dir dir, const char* msArg) {
  long ms = 0;
  if (msArg && *msArg) ms = atol(msArg);
  if (ms < 0) ms = 0;
  if ((unsigned long)ms > MAX_RUN_MS) ms = MAX_RUN_MS;

  setDir(m, dir);
  if (ms > 0) { m.timed = true;  m.runUntil = millis() + (unsigned long)ms; }
  else        { m.timed = false; m.runUntil = 0; }   // continuous until STOP
  replyOK();
}

// Move the stepper `steps` (signed) with `delayMs` between steps. Blocking.
void doStep(long steps, unsigned long delayMs) {
  int dir = (steps >= 0) ? 1 : -1;
  long n = labs(steps);
  if (n > STEP_MAX) n = STEP_MAX;
  for (long i = 0; i < n; i++) {
    g_stepPhase = (g_stepPhase + dir + 8) & 7;
    stepperWritePhase(g_stepPhase);
    g_stepPos += dir;
    serviceMotorTimers();      // keep DC timed-stops honoured during the move
    delay(delayMs);
  }
  stepperRelease();
}

void doStatus() {
  char buf[96];
  int n = 0;
  for (int i = 0; i < NUM_MOTORS && n < (int)sizeof(buf); i++) {
    char d = (motors[i].dir == FORWARD) ? 'F' : (motors[i].dir == REVERSE) ? 'R' : 'S';
    n += snprintf(buf + n, sizeof(buf) - n, "M%d:%c ", i + 1, d);
  }
  n += snprintf(buf + n, sizeof(buf) - n, "STEP:%ld ", g_stepPos);
  snprintf(buf + n, sizeof(buf) - n, "ESTOP%d", g_estopped ? 1 : 0);
  replyOK(buf);
}

// --------------------------- command dispatch --------------------------------
void handleLine(char* line) {
  char* cmd = strtok(line, " ");
  if (!cmd) return;
  char* a1 = strtok(nullptr, " ");
  char* a2 = strtok(nullptr, " ");

  if (!strcmp(cmd, "PING")) {
    replyOK("PONG");
  } else if (!strcmp(cmd, "STATUS")) {
    doStatus();
  } else if (!strcmp(cmd, "STOP")) {
    if (a1 && *a1) {
      Motor* m = motorArg(a1);
      if (!m) replyErr("STOP expects a motor 1..N, or nothing for all");
      else   { stopMotor(*m); replyOK(); }
    } else {
      stopAll();
      replyOK();
    }
  } else if (!strcmp(cmd, "ESTOP")) {
    stopAll();
    g_estopped = true;
    replyOK("ESTOP");
  } else if (!strcmp(cmd, "CLEAR")) {
    g_estopped = false;
    replyOK();
  } else if (g_estopped) {
    replyErr("estopped; send CLEAR first");
  } else if (!strcmp(cmd, "FWD") || !strcmp(cmd, "REV")) {
    Motor* m = motorArg(a1);
    if (!m) {
      replyErr("FWD/REV expects a motor 1..N");
    } else if (m->pinB < 0 && !strcmp(cmd, "REV")) {
      replyErr("single-relay motor: only FWD/STOP");
    } else {
      startRun(*m, !strcmp(cmd, "FWD") ? FORWARD : REVERSE, a2);
    }
  } else if (!strcmp(cmd, "STEP")) {
    if (!a1 || !*a1) {
      replyErr("STEP expects a signed step count");
    } else {
      unsigned long d = (a2 && *a2) ? (unsigned long)atol(a2) : STEP_DELAY_MS;
      doStep(atol(a1), d);
      replyOK();
    }
  } else {
    replyErr("unknown command");
  }
}

// ------------------------------- setup/loop ----------------------------------
void setup() {
  // Relays: drive de-energized BEFORE enabling outputs, so nothing clicks on.
  for (int i = 0; i < NUM_MOTORS; i++) {
    relayWrite(motors[i].pinA, false);
    if (motors[i].pinB >= 0) relayWrite(motors[i].pinB, false);
    pinMode(motors[i].pinA, OUTPUT);
    if (motors[i].pinB >= 0) pinMode(motors[i].pinB, OUTPUT);
    relayWrite(motors[i].pinA, false);
    if (motors[i].pinB >= 0) relayWrite(motors[i].pinB, false);
  }
  // Stepper: release all coils.
  pinMode(STEP_IN1, OUTPUT); pinMode(STEP_IN2, OUTPUT);
  pinMode(STEP_IN3, OUTPUT); pinMode(STEP_IN4, OUTPUT);
  stepperRelease();

  Serial.begin(115200);
  Serial.println("# esp32_dc_prototype ready");
}

void loop() {
  serviceMotorTimers();

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
