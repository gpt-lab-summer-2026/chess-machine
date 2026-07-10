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
#define R_FWD_PIN      22    // FORWARD = +r = cart OUTWARD  (energize pin 22 to drive out)
#define R_REV_PIN      23    // REVERSE = -r = cart TOWARD   (energize pin 23 to drive toward)
// A axis = rotating base, 2-relay H-bridge (relay "2"). Swap FWD/REV to invert.
#define A_FWD_PIN       4    // energize to rotate CCW / toward +degrees  [proto 2 f]
#define A_REV_PIN      15    // energize to rotate CW  / toward -degrees  [proto 2 r]
// Winch = 28BYJ-48 via ULN2003. Coils IN1..IN4 (proto order; IN2<->IN3 already
// swapped so it rotates instead of vibrating). IN4 is 17, NOT 3 (that's UART RX).
#define P_IN1           5
#define P_IN2          21
#define P_IN3          18
#define P_IN4          17
// Endstops: INPUT_PULLUP, switch to GND, LOW = pressed. NONE are wired on the
// current build — these are placeholders. Pin 15 is now the A-axis reverse relay,
// so the top endstop moved off 15 to avoid a conflict.
#define R_ENDSTOP_PIN  13    // radial (inner end of the rail)      [not wired yet]
#define A_ENDSTOP_PIN  14    // rotary home switch                  [not wired yet]
#define P_TOP_ENDSTOP  26    // winch fully-retracted (top) switch  [not wired yet]
#define MAGNET_PIN     19    // relay "1" for the electromagnet     [proto 1 / single relay]

// ================ EDIT: relay polarity / dead-time ===========================
const bool          RELAY_ACTIVE_LOW = true;  // most hobby relay boards: LOW = energized
const bool          MAGNET_ACTIVE_LOW = false;// magnet relay/MOSFET: false = HIGH energizes
const unsigned long RELAY_SETTLE_MS  = 30;    // dead-time when reversing an H-bridge
const bool          HAS_DC_ENDSTOPS  = false; // no radial/rotary switches wired yet

// ================ EDIT: DC calibration (from bench measurements) =============
// Linear cart, full 323 mm travel timed PER DIRECTION; the deadzone is ALREADY
// included in these times, so we divide straight through and set deadzone = 0.
// OUT 1150 ms = FORWARD (+r); IN 950 ms = REVERSE (-r) (it reels in faster).
const float         R_MS_PER_MM_OUT = 1150.0f / 323.0f;  // ~3.56 ms/mm (FORWARD / +r / out)
const float         R_MS_PER_MM_IN  =  950.0f / 323.0f;  // ~2.94 ms/mm (REVERSE / -r / toward)
const unsigned long R_DEADZONE_MS   = 0;   // already folded into the times above
// Rotary base: 360 deg turn = 8650 ms, same both ways (deadzone included).
const float         A_MS_PER_DEG    = 8650.0f / 360.0f;  // ~24.0 ms/deg
const unsigned long A_DEADZONE_MS   = 0;

// ================ EDIT: soft limits & homing =================================
const float R_MIN_MM   = 80.0f;    // reachable annulus (rail is mechanically longer)
const float R_MAX_MM   = 300.0f;
const float A_MIN_DEG  = -55.0f;   // reachable sweep
const float A_MAX_DEG  = 55.0f;
const float R_HOME_MM  = 80.0f;    // radius of the PARK pose (cart fully in, = r_min)
const float A_HOME_DEG = 0.0f;     // pivot-frame angle of PARK pose (arm at board centre)
const float R_HOME_BACKOFF_MM  = 3.0f;   // back off the switch after homing
const float A_HOME_BACKOFF_DEG = 3.0f;
const unsigned long HOME_SEEK_TIMEOUT_MS = 20000;  // give up seeking a dead switch

// ================ EDIT: winch (28BYJ-48 / ULN2003) ===========================
// Cable moved per output revolution = circumference = 2*pi*drum radius, and the
// 28BYJ-48 does ~4096 half-steps per output revolution (half-step drive), so
// steps/mm = 4096 / (2*pi*radius).
const float         WINCH_DRUM_RADIUS_MM = 7.0f;     // spool radius the cable winds on (0.7 cm drum)
const float         WINCH_STEPS_PER_REV  = 4096.0f;  // 28BYJ-48 half-steps per output rev
const float         P_STEPS_PER_MM   = WINCH_STEPS_PER_REV / (TWO_PI * WINCH_DRUM_RADIUS_MM);  // ~93
const int           P_UP_STEP_DIR    = +1;    // step sign that RAISES the magnet
const unsigned long WINCH_STEP_DELAY_MS = 2;  // per half-step (speed)
const bool          P_HAS_TOP_ENDSTOP = false;  // no top switch wired — HOME won't seek
const float         P_MAX_HEIGHT_MM   = 80.0f;// max lift / HOME parked height (small drum -> keep the lift modest)
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
  float         msPerFwd;      // travel time per unit in the + (FWD) direction
  float         msPerRev;      // travel time per unit in the - (REV) direction
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
  "radial(cart)", R_FWD_PIN, R_REV_PIN, R_MS_PER_MM_OUT, R_MS_PER_MM_IN, R_DEADZONE_MS,
  R_MIN_MM, R_MAX_MM, R_ENDSTOP_PIN, REVERSE, R_HOME_MM, R_HOME_BACKOFF_MM,
  STOPPED, R_HOME_MM, false, R_HOME_MM, 0,
};
DcAxis aAxis = {
  "rotary(base)", A_FWD_PIN, A_REV_PIN, A_MS_PER_DEG, A_MS_PER_DEG, A_DEADZONE_MS,
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
  float rate = (d == FORWARD) ? ax.msPerFwd : ax.msPerRev;
  unsigned long dur = ax.deadzoneMs + (unsigned long)(fabs(delta) * rate + 0.5f);
  dcSetDir(ax, d);
  ax.endTime = millis() + dur;
  ax.moving = true;
}

// Advance one axis's timed run: stop on its deadline, or early if it drives into
// its endstop (overrun backstop — the host normally stays inside the soft limits).
void dcService(DcAxis& ax) {
  if (!ax.moving) return;
  if (HAS_DC_ENDSTOPS && ax.dir == ax.homeSeekDir && endstopPressed(ax.endstopPin)) {
    dcStop(ax);
    ax.cur = ax.homeValue;
    return;
  }
  if ((long)(millis() - ax.endTime) >= 0) {
    dcStop(ax);
    ax.cur = ax.target;
  }
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
// NO ENDSTOPS on this build, so HOME does NOT move anything — blindly seeking a
// switch that isn't there is exactly what drove the crane off on startup.
// Instead, HOME declares the crane's CURRENT physical pose to be the reference.
// PARK THE CRANE BY HAND at that pose FIRST, then send HOME:
//   * arm pointing at the board centre   -> A_HOME_DEG (0 deg)
//   * cart fully in against inner stop   -> R_HOME_MM (r_min)
//   * magnet wound up to the top         -> P_MAX_HEIGHT_MM
// When you add real endstops, set HAS_DC_ENDSTOPS / P_HAS_TOP_ENDSTOP and restore
// a seeking homer (see git history) to automate this.
void doHome() {
  g_estopped = false;
  dcStop(rAxis);
  dcStop(aAxis);
  rAxis.cur = R_HOME_MM;
  aAxis.cur = A_HOME_DEG;
  g_curH    = P_MAX_HEIGHT_MM;
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
