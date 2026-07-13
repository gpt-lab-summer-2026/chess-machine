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
// A axis = rotating base — NOW A STEPPER (28BYJ-48 via ULN2003), coils IN1..IN4.
// (Was a DC H-bridge on pins 4/15, now freed. The DC base under-rotated with
//  backlash; a stepper turns by exact step count, so that error is gone.)
#define A_IN1          27
#define A_IN2          26
#define A_IN3          25
#define A_IN4          33

#define echoPin 12
#define trigPin 13 
// Winch = 28BYJ-48 via ULN2003. Coils IN1..IN4 (proto order; IN2<->IN3 already
// swapped so it rotates instead of vibrating). IN4 is 17, NOT 3 (that's UART RX).
#define P_IN1           5
#define P_IN2          21
#define P_IN3          18 
#define P_IN4          17
// Endstops: NONE wired. Pins 26 & 33 are now the base stepper (A_IN2/A_IN4), so
// the old top/rotary endstop defines are gone; only R keeps a placeholder pin.
#define MAGNET_PIN     19    // relay "1" for the electromagnet     [proto 1 / single relay]
#define R_ENDSTOP_PIN  32    // radial inner switch  [not wired]


#define SOUND_SPEED 0.034

// ================ EDIT: ultrasoun
// Down-looking sensor: ~39 cm to the bare table, CLOSER over a box placed at the
// diagonal home. Sweep the boom until it sees the box, then call that A_HOME_DEG.
const unsigned long US_TIMEOUT_US        = 20000;  // pulseIn cap (us) — never block the loop
const float         US_HOME_THRESHOLD_CM = 10.0f;  // reading below this = box below = home
const unsigned long US_PING_GAP_MS       = 20;     // settle between pings during the sweep
const float         US_SEEK_DEG          = 45.0f;  // sweep this far each way to find the box
const unsigned long US_PRINT_MS          = 500;    // stream the reading as a '#' line every N ms (0 = off)

// ================ EDIT: relay polarity / dead-time ===========================
const bool          RELAY_ACTIVE_LOW = true;  // most hobby relay boards: LOW = energized
const bool          MAGNET_ACTIVE_LOW = false;// magnet relay/MOSFET: false = HIGH energizes
const unsigned long RELAY_SETTLE_MS  = 30;    // dead-time when reversing an H-bridge
const unsigned long AXIS_STAGGER_MS  = 60;    // gap between axes: only ONE DC motor runs at a time (avoids dual inrush -> brownout)
const bool          HAS_DC_ENDSTOPS  = false; // no radial/rotary switches wired yet

// ================ EDIT: DC calibration (from bench measurements) =============
// Each measured full-travel time INCLUDES a fixed startup dead-time (relay
// settle + static friction) during which the motor hasn't begun moving yet. We
// separate it out so the command time is  deadzone + distance*rate  and SHORT
// moves aren't undershot. (Baking the deadzone into a linear ms/mm undershoots
// every short move; that open-loop error accumulates until the tracked position
// drifts past the physical cart and the axis drives INWARD into the stop — the
// gear-grinding bug this fixes.)  These are tunable — measure the dead-time as
// the delay between energizing and first visible motion.
// Linear cart, full 320 mm travel (end to end) timed PER DIRECTION.
// OUT 1150 ms = FORWARD (+r); IN 950 ms = REVERSE (-r) (it reels in faster).
const unsigned long R_DEADZONE_MS   = 30;    // startup dead-time, both directions
// Loaded correction: on the real arm the cart UNDER-extends ~20-25% vs these
// UNLOADED bench times (it moves slower under load). The corners test showed the
// far corners reaching only ~0.79 of the commanded radius, so scale the rate up.
// Tune this factor from the corners test, or re-measure loaded travel with JOG.
const float         R_LOAD_FACTOR   = 1.25f;
const float         R_MS_PER_MM_OUT = R_LOAD_FACTOR * (1150.0f - R_DEADZONE_MS) / 320.0f;  // ~4.37 ms/mm (FWD / +r / out)
const float         R_MS_PER_MM_IN  = R_LOAD_FACTOR * ( 950.0f - R_DEADZONE_MS) / 320.0f;  // ~3.59 ms/mm (REV / -r / in)
// Rotary base: STEPPER (28BYJ-48 via ULN2003) — exact step count, so no undershoot
// or backlash (the reason we switched off the DC base). steps/deg = motor half-steps
// per rev (4096) * gear ratio / 360. PLACEHOLDER assumes DIRECT drive (11.38); if the
// base is geared down this is much higher -- CALIBRATE with `JOG A<steps>`: rotate a
// known step count, measure the swept angle, steps/deg = steps / degrees.
const float         A_STEPS_PER_DEG = 11.38f;  // CALIBRATE (x gear ratio if geared)
const int           A_STEP_DIR      = +1;      // step sign for +degrees; flip to invert
const unsigned long A_STEP_DELAY_MS = 2;       // per half-step (speed)

// ================ EDIT: soft limits & homing =================================
// R is the CART's radial position from the pivot (serial_esp32 sends the cart
// target, not the magnet's). At the inner stop the cart hasn't traveled at all,
// but the winch/magnet mount sits ~mid-cart, so the cart reference is already
// out past the bare crane structure: 80 mm structure + ~37.5 mm (winch ~half of
// the 75 mm cart) = ~117.5 mm. THAT is r_min / home, NOT the 80 mm structure —
// using 80 made every return over-reel ~37 mm into the inner stop.
const float R_MIN_MM   = 117.5f;   // cart R at the inner mechanical stop (= home)
const float R_MAX_MM   = 410.0f;   // a1 needs ~379 mm; cart travel is 320 mm so the far stop is 117.5+320=437.5 — stay under it
const float A_MIN_DEG  = -55.0f;   // reachable sweep
const float A_MAX_DEG  = 55.0f;
const float R_HOME_MM  = 117.5f;   // PARK pose = cart fully in against the inner stop (= r_min)
const float A_HOME_DEG = 0.0f;     // pivot-frame angle of PARK pose (arm at board centre)
const float R_HOME_BACKOFF_MM  = 3.0f;   // legacy (unused without endstops)
const float A_HOME_BACKOFF_DEG = 3.0f;   // legacy (unused without endstops)
// HOME re-homes PHYSICALLY (no endstops wired): the rail seeks its INNER hard
// stop by driving inward for the time to cover the tracked distance from home
// (current position x reverse speed) plus a short seating push, capped at a hard
// max so a corrupt tracked value can never run the motor indefinitely.
const unsigned long R_HOME_SEAT_MS     = 120;    // extra inward push to seat against the stop
const unsigned long R_HOME_SEEK_MAX_MS = 1100;   // hard cap (~920 ms for the full 320 mm + seat)
const unsigned long JOG_MAX_MS         = 5000;   // safety cap for the manual JOG calibration command

// ================ EDIT: winch (28BYJ-48 / ULN2003), TWO-POSITION =============
// The winch has exactly two working positions a fixed step count apart:
//   TRAVEL (up)   = where it starts and rests between every move
//   PICK   (down) = magnet lowered onto a piece
// Measured stroke = 7425 half-steps: -7425 lowers to PICK, +7425 climbs to TRAVEL.
// It ALWAYS begins at TRAVEL and returns to TRAVEL after each move. (This replaces
// the old mm x steps/mm math; the bigger spool made a direct measurement simpler.)
const long          WINCH_STROKE_STEPS  = 8050;  // half-steps between TRAVEL and PICK
const int           P_UP_STEP_DIR       = +1;    // step sign that RAISES the magnet (flip to invert)
const unsigned long WINCH_STEP_DELAY_MS = 2;     // per half-step (speed)
const float         WINCH_PICK_BELOW_MM = 30.0f; // a PULLEY height below this = drop to PICK
const bool          P_HAS_TOP_ENDSTOP   = false; // no top switch wired — HOME won't seek
const float         P_MAX_HEIGHT_MM     = 80.0f; // nominal TRAVEL/parked height (STATUS + HOME)
const unsigned long WINCH_HOME_TIMEOUT_MS = 20000;
// ==================================================
//===========================

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
  float         Echo;
};

DcAxis rAxis = {
  "radial(cart)", R_FWD_PIN, R_REV_PIN, R_MS_PER_MM_OUT, R_MS_PER_MM_IN, R_DEADZONE_MS,
  R_MIN_MM, R_MAX_MM, R_ENDSTOP_PIN, REVERSE, R_HOME_MM, R_HOME_BACKOFF_MM,
  STOPPED, R_HOME_MM, false, R_HOME_MM, 0,
};
// The rotating base is no longer a DcAxis — it's a STEPPER, tracked by g_curA below.

bool    g_estopped = false;
bool    g_magnetOn = false;
float   g_curH     = P_MAX_HEIGHT_MM;  // tracked winch height (mm); boot assumes the magnet is parked at the top
bool    g_winchAtPick = false;         // two-position winch: false = TRAVEL (up, start), true = PICK (down)
int     g_stepPhase = 0;     // winch half-step phase index
float   g_curA     = A_HOME_DEG;       // tracked base angle (deg); the base is a step-counted stepper
int     g_aStepPhase = 0;              // base stepper half-step phase index
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
  float delta = target - ax.cur;
  // Safety: never drive INWARD once we're at the inner mechanical stop (the
  // crane deadzone). No endstops are wired, so an open-loop reverse here just
  // grinds the gears against the stop. Refuse it and keep the tracked position.
  if (delta < 0.0f && ax.cur <= ax.minLimit + MOVE_EPS) {
    ax.target = ax.cur;
    dcStop(ax);
    return;
  }
  ax.target = target;
  if (fabs(delta) < MOVE_EPS) { dcStop(ax); return; }
  Dir d = (delta < 0) ? FORWARD : REVERSE;
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

// ----------------------------- base (stepper) --------------------------------
void baseWritePhase(int phase) {
  digitalWrite(A_IN1, HALFSTEP[phase][0]);
  digitalWrite(A_IN2, HALFSTEP[phase][1]);
  digitalWrite(A_IN3, HALFSTEP[phase][2]);
  digitalWrite(A_IN4, HALFSTEP[phase][3]);
}
void baseRelease() {   // de-energize all base coils (ESTOP / boot)
  digitalWrite(A_IN1, LOW); digitalWrite(A_IN2, LOW);
  digitalWrite(A_IN3, LOW); digitalWrite(A_IN4, LOW);
}
// Rotate the base `steps` half-steps (signed). Blocking; leaves coils energized so
// the base holds its bearing against any load.
void baseStep(long steps) {
  int dir = (steps >= 0) ? 1 : -1;
  long n = labs(steps);
  for (long i = 0; i < n; i++) {
    g_aStepPhase = (g_aStepPhase + dir + 8) & 7;
    baseWritePhase(g_aStepPhase);
    delay(A_STEP_DELAY_MS);
  }
}
// Rotate the base to absolute angle `adeg` (clamped to the soft sweep). Exact.
void baseMoveTo(float adeg) {
  adeg = clampf(adeg, A_MIN_DEG, A_MAX_DEG);
  long steps = lroundf((adeg - g_curA) * A_STEPS_PER_DEG) * A_STEP_DIR;
  baseStep(steps);
  g_curA = adeg;
}

// ----------------------------- homing ----------------------------------------
// NO ENDSTOPS on this build, so HOME re-homes PHYSICALLY by timing instead of
// reading a switch. It (1) raises the magnet clear, (2) swings the arm back to
// the park bearing, and (3) drives the rail into its inner hard stop for a true
// R zero. Steps 1-2 use the tracked position (open-loop); only the rail gets a
// real mechanical reference. Add a rotary / winch endstop to zero those for real.
//
// Timed seek to the rail's inner stop: drive inward long enough to cover the
// tracked distance from home (current position x reverse speed) plus a seating
// push, capped so a bad tracked value can't run the motor forever. The cart then
// rests against the stop (relays de-energized), so R is genuinely re-zeroed.
void homeRail() {
  float dist = rAxis.cur - R_HOME_MM;
  if (dist < 0.0f) dist = 0.0f;
  unsigned long t = (unsigned long)(dist * R_MS_PER_MM_IN + 0.5f) + R_HOME_SEAT_MS;
  if (t > R_HOME_SEEK_MAX_MS) t = R_HOME_SEEK_MAX_MS;
  char buf[96];
  snprintf(buf, sizeof(buf), "# HOME rail: seek IN from R%.2f, %.2f mm -> %lu ms (seat %lu, cap %lu)",
           rAxis.cur, dist, t, R_HOME_SEAT_MS, R_HOME_SEEK_MAX_MS);
  Serial.println(buf);
  dcSetDir(rAxis, FORWARD);
  unsigned long end = millis() + t;
  while ((long)(millis() - end) < 0) { /* drive inward into the stop */ }
  dcStop(rAxis);
  rAxis.cur = R_HOME_MM;
}

// Calibration jog: drive one DC axis for a FIXED time (signed ms: + = FWD/out,
// - = REV/in), bypassing positioning + soft limits. Does NOT update the tracked
// position, so send HOME afterwards. Use it to measure real loaded speed: jog a
// known time, measure the travel, then ms/mm = time / distance.
void jogAxis(DcAxis& ax, long ms) {
  Dir d = (ms >= 0) ? FORWARD : REVERSE;
  unsigned long t = (unsigned long)labs(ms);
  if (t > JOG_MAX_MS) t = JOG_MAX_MS;
  char buf[96];
  snprintf(buf, sizeof(buf), "# JOG %s %s %lu ms (tracked position now UNKNOWN -> send HOME after)",
           ax.name, (d == FORWARD) ? "FWD/out" : "REV/in", t);
  Serial.println(buf);
  dcSetDir(ax, d);
  unsigned long end = millis() + t;
  while ((long)(millis() - end) < 0) { /* run for the fixed jog time */ }
  dcStop(ax);
}

// Drive one DC axis to an absolute target, blocking until it stops. Only this
// axis is serviced, so no other motor is energized while it runs.
void driveAxisBlocking(DcAxis& ax, float target) {
  dcStartMove(ax, target);
  while (ax.moving) dcService(ax);
}

// --- ultrasound rotary homing --------------------------------------------------
// One reading from the down-looking ultrasound (HC-SR04). Bounded pulseIn so a
// missing echo can never block the control loop; returns a large value on timeout.
float readUltrasoundCm() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  unsigned long dur = pulseIn(echoPin, HIGH, US_TIMEOUT_US);
  if (dur == 0) return 999.0f;             // no echo within the cap = treat as far/clear
  return dur * SOUND_SPEED / 2.0f;
}

// Step the base up to `deg` in direction `dir` (+1/-1), pinging the ultrasound
// every ~1 deg; stop and return true the instant the box appears below the sensor.
bool seekBoxSweep(int dir, float deg) {
  long total = lroundf(deg * A_STEPS_PER_DEG);
  long chunk = lroundf(A_STEPS_PER_DEG);
  if (chunk < 1) chunk = 1;                       // ~1 deg per ping
  for (long done = 0; done < total; done += chunk) {
    long n = (total - done < chunk) ? (total - done) : chunk;
    baseStep(n * dir * A_STEP_DIR);
    if (readUltrasoundCm() < US_HOME_THRESHOLD_CM) return true;
    delay(US_PING_GAP_MS);
  }
  return false;
}

// Rotary home via ultrasound: sweep +/-US_SEEK_DEG until the down-looking sensor
// sees the home box (closer than the bare table), then declare that bearing
// A_HOME_DEG. Leaves A on its tracked estimate if the box is never found.
bool homeRotaryUS() {
  Serial.println("# HOME rotary: ultrasound box seek (stepper)");
  bool found = (readUltrasoundCm() < US_HOME_THRESHOLD_CM);   // already over the box?
  if (!found) found = seekBoxSweep(+1, US_SEEK_DEG) || seekBoxSweep(-1, 2.0f * US_SEEK_DEG);
  if (found) { g_curA = A_HOME_DEG; Serial.println("# HOME rotary: box found -> A zeroed"); }
  else       { Serial.println("# HOME rotary: box NOT found (kept tracked A)"); }
  return found;
}

void doHome() {
  g_estopped = false;
  dcStop(rAxis);
  char buf[96];
  snprintf(buf, sizeof(buf), "# HOME start: from R%.2f A%.2f H%.2f", rAxis.cur, g_curA, g_curH);
  Serial.println(buf);
  // 1) Raise the magnet to the top so the sweep clears the board. The winch is a
  //    stepper (holds its count), so this is an accurate absolute move.
  doPulley(P_MAX_HEIGHT_MM);
  // 2) Rotary: find the home box with the down-looking ultrasound (a real bearing
  //    reference now, not open-loop). Sweeps +/-US_SEEK_DEG; on the box it zeroes A
  //    to home, else it keeps the tracked angle.
  homeRotaryUS();
  delay(AXIS_STAGGER_MS);         // let the base motor settle before the rail runs
  // 3) Rail: physically seek the inner mechanical stop for a true R zero.
  homeRail();
  snprintf(buf, sizeof(buf), "# HOME done: R%.2f A%.2f H%.2f (rail seated at inner stop)",
           rAxis.cur, g_curA, g_curH);
  Serial.println(buf);
}

// ----------------------------- motion ----------------------------------------
// Move both DC axes to absolute (rmm, adeg) ONE AT A TIME — never together — so
// only a single motor ever draws current. Two simultaneous inrush spikes were
// sagging the supply and browning out the board (relays then float -> runaway).
// Rotate first (cart retracted = smaller swing), let the base current settle,
// then extend the cart. Blocks until both finish; `feed` is ignored.
void doMoveRA(float rmm, float adeg) {
  baseMoveTo(adeg);               // rotating base: STEPPER, exact angle
  delay(AXIS_STAGGER_MS);
  driveAxisBlocking(rAxis, rmm);  // radial cart: DC, timed
}

void doPulley(float hmm) {
  // Two-position winch: a height below WINCH_PICK_BELOW_MM means "drop to PICK",
  // anything above means "climb to TRAVEL". Move the fixed stroke only when the
  // position actually changes; +stroke raises, -stroke lowers.
  bool wantPick = (hmm < WINCH_PICK_BELOW_MM);
  if (wantPick != g_winchAtPick) {
    long steps = WINCH_STROKE_STEPS * P_UP_STEP_DIR * (wantPick ? -1 : +1);
    winchStep(steps, WINCH_STEP_DELAY_MS);
    g_winchAtPick = wantPick;
  }
  g_curH = hmm;   // track the requested height for STATUS
}

void doStatus() {
  char buf[80];
  snprintf(buf, sizeof(buf), "R%.2f A%.2f H%.2f MAG%d ENDR%d ENDA%d",
           rAxis.cur, g_curA, g_curH, g_magnetOn ? 1 : 0,
           endstopPressed(R_ENDSTOP_PIN) ? 1 : 0, 0);   // no A endstop (base is a stepper)
  replyOK(buf);
}

void doEstop() {
  g_estopped = true;
  dcStop(rAxis);
  baseRelease();
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
  } else if (!strcmp(cmd, "USDIST")) {
    char b[32];
    snprintf(b, sizeof(b), "US %.1f", readUltrasoundCm());   // one-shot sensor test
    replyOK(b);
  } else if (!strcmp(cmd, "ESTOP")) {
    doEstop();
  } else if (g_estopped) {
    replyErr("estopped; send HOME to clear");
  } else if (!strcmp(cmd, "MOVE")) {
    float r = rAxis.cur;
    float a = g_curA;
    argKeyed('R', &r);
    argKeyed('A', &a);   // F is accepted but ignored (relays have no speed control)
    doMoveRA(r, a);
    replyOK();
  } else if (!strcmp(cmd, "JOG")) {
    // Bench calibration. R takes a FIXED TIME in ms (DC cart, "JOG R2000"); A takes
    // a signed STEP COUNT (base stepper, "JOG A2048" — rotate, measure the angle,
    // A_STEPS_PER_DEG = steps/deg). Both leave the tracked position UNKNOWN -> HOME.
    float v;
    if (argKeyed('R', &v)) jogAxis(rAxis, (long)v);
    if (argKeyed('A', &v)) baseStep((long)v * A_STEP_DIR);
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
// Why the ESP32 last reset. A BROWNOUT (or an unexpected POWERON mid-session)
// means the motors are sagging the supply and rebooting the board — which cuts a
// move short and floats the relay pins on the way back up.
const char* resetReasonStr() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:   return "POWERON";
    case ESP_RST_EXT:       return "EXT";
    case ESP_RST_SW:        return "SW";
    case ESP_RST_PANIC:     return "PANIC/crash";
    case ESP_RST_INT_WDT:   return "INT_WDT";
    case ESP_RST_TASK_WDT:  return "TASK_WDT";
    case ESP_RST_WDT:       return "WDT";
    case ESP_RST_BROWNOUT:  return "BROWNOUT";
    case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
    default:                return "UNKNOWN";
  }
}

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

  setupDcAxis(rAxis);              // radial cart is still a DC H-bridge

  pinMode(P_IN1, OUTPUT); pinMode(P_IN2, OUTPUT);   // winch stepper
  pinMode(P_IN3, OUTPUT); pinMode(P_IN4, OUTPUT);
  pinMode(A_IN1, OUTPUT); pinMode(A_IN2, OUTPUT);   // base stepper
  pinMode(A_IN3, OUTPUT); pinMode(A_IN4, OUTPUT);

  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input

  stepperRelease();   // winch coils off
  baseRelease();      // base coils off

  char buf[64];
  snprintf(buf, sizeof(buf), "# esp32_chess ready (reset: %s)", resetReasonStr());
  Serial.println(buf);
}

void loop() {
  // Optional ultrasound telemetry: a rate-limited, BOUNDED read emitted as a '#'
  // debug line (the host ignores '#'), so it can't jam the loop or corrupt the
  // protocol the way the old per-iteration unbounded pulseIn did. It only runs
  // between commands (a blocking MOVE/HOME pauses it). Set US_PRINT_MS = 0 to mute.
  static unsigned long lastUs = 0;
  if (US_PRINT_MS && (long)(millis() - lastUs) >= (long)US_PRINT_MS) {
    lastUs = millis();
    char b[40];
    snprintf(b, sizeof(b), "# DIST %.1f cm", readUltrasoundCm());
    Serial.println(b);
  }

  // Serial command handling.
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
