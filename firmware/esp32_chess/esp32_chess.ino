/*
 * esp32_chess — motor controller firmware for the voice-controlled chess board.
 *
 * Drives a polar crane, but with the REAL hybrid drivetrain (this is a first
 * sketch of the control system for the actual build):
 *
 *   - R axis (radial, mm from the pivot) = the LINEAR CART on the arm.
 *     Brushed DC motor via a 2-relay H-bridge. Positioned by TIME (open-loop).
 *   - A axis (angle, degrees in the pivot frame) = the ROTATING BASE.
 *     Brushed DC motor via a 2-relay H-bridge. Positioned by TIME (open-loop).
 *   - Pulley H (mm) = the WINCH. 28BYJ-48-style stepper via a ULN2003 driver.
 *   - One electromagnet on a single relay.
 *
 * It speaks the SAME line protocol the host (chessmachine.motion.serial_esp32)
 * expects — only the motor layer changed from steppers to timed DC + a winch:
 *
 *   PING                          -> OK PONG
 *   HOME                          -> OK HOMED
 *   MOVE R<mm> A<deg> [F<mm/min>] -> OK
 *   PULLEY H<mm> [F<mm/min>]      -> OK
 *   MAG ON|OFF                    -> OK
 *   STATUS                        -> OK R<f> A<f> H<f> MAG<0|1> ENDR<0|1> ENDA<0|1>
 *   ESTOP                         -> OK ESTOP
 *
 * The host does the Cartesian->polar conversion, so this firmware only positions
 * the radial axis (mm) and the rotary axis (degrees). Every motion command BLOCKS
 * until the move finishes, then replies OK, so the host never has to track motor
 * state. The DC axes are relays (bang-bang), so the feed `F` is accepted and
 * IGNORED — travel time is fixed by each motor's own speed. Lines starting with
 * '#' are debug.
 *
 * Board: any ESP32 dev module. No external libraries. EDIT the pin + calibration
 * section for your wiring / measurements.
 */

// ======================= EDIT: pins ==========================================
// R axis = linear cart, 2-relay H-bridge (relay "3"). Swap FWD/REV to invert.
#define R_FWD_PIN      26    // energize to drive the cart OUTWARD (r increasing)
#define R_REV_PIN      16    // energize to drive the cart INWARD  (r decreasing)
// A axis = rotating base, 2-relay H-bridge (relay "2"). Swap FWD/REV to invert.
#define A_FWD_PIN      25    // energize to rotate toward +degrees
#define A_REV_PIN      27    // energize to rotate toward -degrees
// Winch = 28BYJ-48 via ULN2003. Coils IN1..IN4. If it only VIBRATES, swap IN2<->IN3.
#define P_IN1          33
#define P_IN2          32
#define P_IN3          18
#define P_IN4          19
// Endstops: INPUT_PULLUP, switch to GND, LOW = pressed.
#define R_ENDSTOP_PIN  13    // radial (inner end of the rail)
#define A_ENDSTOP_PIN  14    // rotary home switch
#define P_TOP_ENDSTOP  15    // winch fully-retracted (top) switch
#define MAGNET_PIN     23    // relay "1" for the electromagnet
// GPIO5 is free (was the old shared stepper-driver enable) — reserved for the
// winch's own power-enable relay if/when that gets wired.

// ================ EDIT: relay polarity / dead-time ===========================
const bool          RELAY_ACTIVE_LOW = true;  // most hobby relay boards: LOW = energized
const bool          MAGNET_ACTIVE_LOW = false;// magnet relay/MOSFET: false = HIGH energizes
const unsigned long RELAY_SETTLE_MS  = 30;    // dead-time when reversing an H-bridge

// ================ EDIT: DC calibration (from bench measurements) =============
// Linear cart: 315 mm of travel in 1050 ms, of which 30 ms is start-up deadzone
//   => 315 mm in 1020 ms of motion => 3.238 ms per mm.
const float         R_MS_PER_MM   = 3.238f;
const unsigned long R_DEADZONE_MS = 30;
// Rotary base: 0.1887 rad/s = 10.81 deg/s => 92.5 ms per degree; ~50 ms deadtime.
const float         A_MS_PER_DEG  = 92.5f;
const unsigned long A_DEADZONE_MS = 50;

// ================ EDIT: soft limits & homing =================================
const float R_MIN_MM   = 80.0f;    // reachable annulus (rail is mechanically longer)
const float R_MAX_MM   = 300.0f;
const float A_MIN_DEG  = -55.0f;   // reachable sweep
const float A_MAX_DEG  = 55.0f;
const float R_HOME_MM  = 80.0f;    // radius at the radial endstop (= r_min, inner)
const float A_HOME_DEG = -60.0f;   // pivot-frame angle at the rotary endstop
const float R_HOME_BACKOFF_MM  = 3.0f;   // back off the switch after homing
const float A_HOME_BACKOFF_DEG = 3.0f;
const unsigned long HOME_SEEK_TIMEOUT_MS = 20000;  // give up seeking a dead switch

// ================ EDIT: winch (28BYJ-48 / ULN2003) ===========================
// PLACEHOLDER until the drum is calibrated: steps of the 28BYJ-48 per mm of wire.
const float         P_STEPS_PER_MM   = 80.0f;
const int           P_UP_STEP_DIR    = +1;    // step sign that RAISES the magnet
const unsigned long WINCH_STEP_DELAY_MS = 2;  // per half-step (speed)
const bool          P_HAS_TOP_ENDSTOP = true;
const float         P_MAX_HEIGHT_MM   = 80.0f;// magnet height when the top switch trips
const unsigned long WINCH_HOME_TIMEOUT_MS = 20000;
// =============================================================================

const float MOVE_EPS = 0.05f;   // ignore sub-this deltas (no motor twitch)

// 28BYJ-48 half-step sequence (IN1..IN4). Reverse = walk it backwards.
const uint8_t HALFSTEP[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1},
};

enum Dir { STOPPED, FORWARD, REVERSE };

// A timed, open-loop DC axis driven through a 2-relay SPDT H-bridge.
struct DcAxis {
  const char*   name;
  int           pinFwd;        // relay energizing the +coordinate direction
  int           pinRev;        // relay energizing the -coordinate direction
  float         msPerUnit;     // travel time per mm (R) or per degree (A)
  unsigned long deadzoneMs;    // fixed lag before motion on every start
  float         minLimit;
  float         maxLimit;
  int           endstopPin;
  Dir           homeSeekDir;   // direction that drives toward the endstop
  float         homeValue;     // coordinate at the endstop
  float         homeBackoff;   // units to back off the switch after homing
  // --- runtime state ---
  Dir           dir;
  float         cur;           // tracked absolute position (open-loop)
  bool          moving;
  float         target;
  unsigned long endTime;
};

DcAxis rAxis = {
  "radial(cart)", R_FWD_PIN, R_REV_PIN, R_MS_PER_MM, R_DEADZONE_MS,
  R_MIN_MM, R_MAX_MM, R_ENDSTOP_PIN, REVERSE, R_HOME_MM, R_HOME_BACKOFF_MM,
  STOPPED, R_HOME_MM, false, R_HOME_MM, 0,
};
DcAxis aAxis = {
  "rotary(base)", A_FWD_PIN, A_REV_PIN, A_MS_PER_DEG, A_DEADZONE_MS,
  A_MIN_DEG, A_MAX_DEG, A_ENDSTOP_PIN, REVERSE, A_HOME_DEG, A_HOME_BACKOFF_DEG,
  STOPPED, A_HOME_DEG, false, A_HOME_DEG, 0,
};

bool    g_estopped = false;
bool    g_magnetOn = false;
float   g_curH     = 0.0f;   // tracked winch height (mm)
int     g_stepPhase = 0;     // current index into HALFSTEP
char    g_line[96];          // line currently being assembled
char    g_argline[96];       // clean copy of the last full line, for arg parsing
uint8_t g_len = 0;

// ----------------------------- helpers ---------------------------------------
inline bool endstopPressed(int pin) { return digitalRead(pin) == LOW; }

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void relayWrite(int pin, bool energized) {
  digitalWrite(pin, (energized != RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

void magnetWrite(bool on) {
  digitalWrite(MAGNET_PIN, (on != MAGNET_ACTIVE_LOW) ? HIGH : LOW);
}

void replyOK(const char* extra = nullptr) {
  if (extra && *extra) { Serial.print("OK "); Serial.println(extra); }
  else                 { Serial.println("OK"); }
}
void replyErr(const char* msg) { Serial.print("ERR "); Serial.println(msg); }

// Find a token like "R12.34" in a space-delimited (mutable) string.
bool readKeyed(char* str, char key, float* out) {
  for (char* tok = strtok(str, " "); tok; tok = strtok(nullptr, " ")) {
    if (tok[0] == key && tok[1] != '\0') { *out = atof(tok + 1); return true; }
  }
  return false;
}
// readKeyed is destructive, so each key gets a fresh copy of g_argline.
bool argKeyed(char key, float* out) {
  char tmp[96];
  strncpy(tmp, g_argline, sizeof(tmp));
  tmp[sizeof(tmp) - 1] = '\0';
  return readKeyed(tmp, key, out);
}

// ----------------------------- DC H-bridge -----------------------------------
// Set an axis's relays for a direction, passing through STOP with a dead-time on
// any change so the bridge never shorts and the motor coasts through zero.
void dcSetDir(DcAxis& ax, Dir d) {
  if (d == ax.dir) return;
  relayWrite(ax.pinFwd, false);
  relayWrite(ax.pinRev, false);
  if (d != STOPPED) delay(RELAY_SETTLE_MS);
  if      (d == FORWARD) relayWrite(ax.pinFwd, true);
  else if (d == REVERSE) relayWrite(ax.pinRev, true);
  ax.dir = d;
}

void dcStop(DcAxis& ax) {
  dcSetDir(ax, STOPPED);
  ax.moving = false;
}

// Arm a timed absolute move; the run itself is serviced by dcService().
void dcStartMove(DcAxis& ax, float target) {
  target = clampf(target, ax.minLimit, ax.maxLimit);
  ax.target = target;
  float delta = target - ax.cur;
  if (fabs(delta) < MOVE_EPS) { dcStop(ax); return; }
  Dir d = (delta > 0) ? FORWARD : REVERSE;
  unsigned long dur = ax.deadzoneMs + (unsigned long)(fabs(delta) * ax.msPerUnit + 0.5f);
  dcSetDir(ax, d);
  ax.endTime = millis() + dur;
  ax.moving = true;
}

// Advance one axis's timed run: stop on its deadline, or early if it drives into
// its endstop (overrun backstop — the host normally stays inside the soft limits).
void dcService(DcAxis& ax) {
  if (!ax.moving) return;
  if (ax.dir == ax.homeSeekDir && endstopPressed(ax.endstopPin)) {
    dcStop(ax);
    ax.cur = ax.homeValue;
    return;
  }
  if ((long)(millis() - ax.endTime) >= 0) {
    dcStop(ax);
    ax.cur = ax.target;
  }
}

// Blocking, un-clamped timed pulse (used for the homing back-off).
void dcTimedPulse(DcAxis& ax, Dir d, float dist) {
  unsigned long dur = ax.deadzoneMs + (unsigned long)(dist * ax.msPerUnit + 0.5f);
  dcSetDir(ax, d);
  unsigned long endT = millis() + dur;
  while ((long)(millis() - endT) < 0) { /* spin */ }
  dcSetDir(ax, STOPPED);
}

// ----------------------------- winch (stepper) -------------------------------
void stepperWritePhase(int phase) {
  digitalWrite(P_IN1, HALFSTEP[phase][0]);
  digitalWrite(P_IN2, HALFSTEP[phase][1]);
  digitalWrite(P_IN3, HALFSTEP[phase][2]);
  digitalWrite(P_IN4, HALFSTEP[phase][3]);
}

void stepperRelease() {   // de-energize all coils (used only on ESTOP / boot)
  digitalWrite(P_IN1, LOW);
  digitalWrite(P_IN2, LOW);
  digitalWrite(P_IN3, LOW);
  digitalWrite(P_IN4, LOW);
}

// Move the winch `steps` (signed) with `delayMs` between half-steps. Blocking.
void winchStep(long steps, unsigned long delayMs) {
  int dir = (steps >= 0) ? 1 : -1;
  long n = labs(steps);
  for (long i = 0; i < n; i++) {
    g_stepPhase = (g_stepPhase + dir + 8) & 7;
    stepperWritePhase(g_stepPhase);
    delay(delayMs);
  }
  // Leave the last phase ENERGIZED so the winch holds its height against the
  // hanging load (no de-energize between moves). Released only on ESTOP / boot.
}

// ----------------------------- homing ----------------------------------------
// Seek the endstop at full speed, stop on contact, call that point `homeValue`,
// then back off the switch into the reachable range.
void homeDcAxis(DcAxis& ax) {
  dcSetDir(ax, ax.homeSeekDir);
  unsigned long deadline = millis() + HOME_SEEK_TIMEOUT_MS;
  while (!endstopPressed(ax.endstopPin) && (long)(millis() - deadline) < 0) { /* seek */ }
  dcSetDir(ax, STOPPED);
  ax.cur = ax.homeValue;
  Dir backDir = (ax.homeSeekDir == REVERSE) ? FORWARD : REVERSE;
  dcTimedPulse(ax, backDir, ax.homeBackoff);
  ax.cur += (backDir == FORWARD ? +ax.homeBackoff : -ax.homeBackoff);
  ax.moving = false;
}

void homeWinch() {
  if (P_HAS_TOP_ENDSTOP) {
    unsigned long deadline = millis() + WINCH_HOME_TIMEOUT_MS;
    while (!endstopPressed(P_TOP_ENDSTOP) && (long)(millis() - deadline) < 0) {
      g_stepPhase = (g_stepPhase + P_UP_STEP_DIR + 8) & 7;   // wind up to the switch
      stepperWritePhase(g_stepPhase);
      delay(WINCH_STEP_DELAY_MS);
    }
    // hold at the top (coils stay energized); released only on ESTOP / boot
  }
  g_curH = P_MAX_HEIGHT_MM;
}

void doHome() {
  g_estopped = false;
  homeDcAxis(rAxis);
  homeDcAxis(aAxis);
  homeWinch();
}

// ----------------------------- motion ----------------------------------------
// Move both DC axes to absolute (rmm, adeg) concurrently, blocking until done.
// `feed` is ignored: relays are bang-bang, so travel time is fixed.
void doMoveRA(float rmm, float adeg) {
  dcStartMove(rAxis, rmm);
  dcStartMove(aAxis, adeg);
  while (rAxis.moving || aAxis.moving) {
    dcService(rAxis);
    dcService(aAxis);
  }
}

void doPulley(float hmm) {
  hmm = clampf(hmm, 0.0f, P_MAX_HEIGHT_MM);
  long steps = lroundf((hmm - g_curH) * P_STEPS_PER_MM) * P_UP_STEP_DIR;
  winchStep(steps, WINCH_STEP_DELAY_MS);
  g_curH = hmm;
}

void doStatus() {
  char buf[80];
  snprintf(buf, sizeof(buf), "R%.2f A%.2f H%.2f MAG%d ENDR%d ENDA%d",
           rAxis.cur, aAxis.cur, g_curH, g_magnetOn ? 1 : 0,
           endstopPressed(R_ENDSTOP_PIN) ? 1 : 0,
           endstopPressed(A_ENDSTOP_PIN) ? 1 : 0);
  replyOK(buf);
}

void doEstop() {
  g_estopped = true;
  dcStop(rAxis);
  dcStop(aAxis);
  stepperRelease();
  magnetWrite(false);
  g_magnetOn = false;
  replyOK("ESTOP");
}

// --------------------------- command dispatch --------------------------------
void handleLine(char* line) {
  char* cmd = strtok(line, " ");   // g_argline still holds the full clean line
  if (!cmd) return;

  if (!strcmp(cmd, "PING")) {
    replyOK("PONG");
  } else if (!strcmp(cmd, "HOME")) {
    doHome();
    replyOK("HOMED");
  } else if (!strcmp(cmd, "STATUS")) {
    doStatus();
  } else if (!strcmp(cmd, "ESTOP")) {
    doEstop();
  } else if (g_estopped) {
    replyErr("estopped; send HOME to clear");
  } else if (!strcmp(cmd, "MOVE")) {
    float r = rAxis.cur;
    float a = aAxis.cur;
    argKeyed('R', &r);
    argKeyed('A', &a);   // F is accepted but ignored (relays have no speed control)
    doMoveRA(r, a);
    replyOK();
  } else if (!strcmp(cmd, "PULLEY")) {
    float h = g_curH;
    argKeyed('H', &h);   // F ignored
    doPulley(h);
    replyOK();
  } else if (!strcmp(cmd, "MAG")) {
    char* arg = strtok(nullptr, " ");
    if (arg && !strcmp(arg, "ON"))       { magnetWrite(true);  g_magnetOn = true;  replyOK(); }
    else if (arg && !strcmp(arg, "OFF")) { magnetWrite(false); g_magnetOn = false; replyOK(); }
    else replyErr("MAG expects ON or OFF");
  } else {
    replyErr("unknown command");
  }
}

// ------------------------------- setup/loop ----------------------------------
void setupDcAxis(DcAxis& ax) {
  // Drive relays de-energized BEFORE enabling outputs, so nothing clicks on.
  relayWrite(ax.pinFwd, false);
  relayWrite(ax.pinRev, false);
  pinMode(ax.pinFwd, OUTPUT);
  pinMode(ax.pinRev, OUTPUT);
  relayWrite(ax.pinFwd, false);
  relayWrite(ax.pinRev, false);
  pinMode(ax.endstopPin, INPUT_PULLUP);
}

void setup() {
  Serial.begin(115200);

  // Magnet off first (GPIO23 has no boot pull-up: add an external pull to the
  // de-energized level if your relay board is active-low).
  magnetWrite(false);
  pinMode(MAGNET_PIN, OUTPUT);
  magnetWrite(false);
  g_magnetOn = false;

  setupDcAxis(rAxis);
  setupDcAxis(aAxis);

  pinMode(P_IN1, OUTPUT); pinMode(P_IN2, OUTPUT);
  pinMode(P_IN3, OUTPUT); pinMode(P_IN4, OUTPUT);
  stepperRelease();
  if (P_HAS_TOP_ENDSTOP) pinMode(P_TOP_ENDSTOP, INPUT_PULLUP);

  Serial.println("# esp32_chess ready");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (g_len > 0) {
        g_line[g_len] = '\0';
        strncpy(g_argline, g_line, sizeof(g_argline));
        g_argline[sizeof(g_argline) - 1] = '\0';
        handleLine(g_line);
        g_len = 0;
      }
    } else if (g_len < sizeof(g_line) - 1) {
      g_line[g_len++] = c;
    }
  }
}
