#include <esp_system.h>  // esp_reset_reason() / ESP_RST_* (not reliably pulled in transitively)

/*
 * esp32_chess — motor controller firmware for the voice-controlled chess board.
 *
 * Drives a polar crane, now with THREE 28BYJ-48 steppers (via ULN2003) + a magnet:
 *
 *   - R axis (radial, mm from the pivot) = the LINEAR CART on the arm. STEPPER
 *     (was a brushed DC motor on a relay H-bridge; swapped to a stepper on the
 *     same gearbox). Step-counted from boot-home = 0 (cart nearest the tower);
 *     a1 is ~18000 steps out at the far end. NEVER slammed into a physical stop
 *     (the stepper can't take it) — the soft limits keep it off both ends.
 *   - A axis (angle, degrees) = the ROTATING BASE. STEPPER, step-counted.
 *   - Pulley H (mm) = the WINCH. STEPPER, two fixed positions (TRAVEL / PICK).
 *   - One electromagnet on a single relay.
 *
 * It speaks the SAME line protocol the host (chessmachine.motion.serial_esp32)
 * expects — the motor layer can change freely underneath it:
 *
 *   PING                          -> OK PONG
 *   HOME                          -> OK HOMED
 *   MOVE R<mm> A<deg> [F<mm/min>] -> OK
 *   GOTO [A<steps>][R<steps>][W<steps>] -> OK  (absolute step-count move; stepmap backend)
 *   PULLEY H<mm> [F<mm/min>]      -> OK
 *   MAG ON|OFF                    -> OK
 *   STATUS                        -> OK R<f> A<f> H<f> MAG<0|1> ENDR<0|1> ENDA<0|1>
 *   ESTOP                         -> OK ESTOP
 *   CAL [ASPD v][AHOME v][RSPM v][AEND v] -> OK CAL ...  (live calibration, no reflash)
 *   JOG [R<steps>][A<steps>]      -> OK           (raw stepper jog, for calibration)
 *   SEEK                          -> OK SEEK AEND<n>  (base: find its limit switch, report the offset)
 *
 * The host does the Cartesian->polar conversion, so this firmware only positions
 * the radial axis (mm) and the rotary axis (degrees). Every motion command BLOCKS
 * until the move finishes, then replies OK. The steppers have no real speed
 * control, so the feed `F` is accepted and IGNORED. Only ONE motor is driven at a
 * time (see AXIS_STAGGER_MS). Lines starting with '#' are debug.
 *
 * The axes are step-counted from a boot-home of 0. The BASE now has a real limit
 * switch (a8 side, backed by a hard stop) it homes against, so it need NOT be
 * hand-placed at boot — set A_ENDSTOP_STEPS to enable it. The winch + cart are still
 * sensorless, so BOOT MUST START WITH the winch at travel/top and the cart at 0
 * (nearest the tower); HOME steps those back to 0 by count. The soft limits
 * (R_MIN/R_MAX, A_MIN/A_MAX) and the base switch keep every move off the physical ends.
 *
 * Board: any ESP32 dev module. No external libraries. EDIT the pin + calibration
 * section for your wiring / measurements.
 */

// ======================= EDIT: pins ==========================================
// R axis = linear CART — STEPPER (28BYJ-48 via ULN2003), coils IN1..IN4.
// (Was a DC H-bridge on 22/23; now a stepper on the same gearbox.)
// NOTE: GPIO 34/35/36/39 are INPUT-ONLY on the ESP32 and CANNOT drive a ULN2003 —
// IN3/IN4 use 4 and 15 (the freed old base-DC pins), which are output-capable.
#define R_IN1 22
#define R_IN2 23
#define R_IN3 13  // was going to be GPIO34 (input-only, won't drive) -> 4
#define R_IN4 14  // was going to be GPIO35 (input-only, won't drive) -> 15
// A axis = rotating base — STEPPER (28BYJ-48 via ULN2003), coils IN1..IN4.
#define A_IN1 27
#define A_IN2 26
#define A_IN3 25
#define A_IN4 33
// Winch = 28BYJ-48 via ULN2003. Coils IN1..IN4 (IN2<->IN3 swapped so it rotates
// instead of vibrating). IN4 is 17, NOT 3 (that's UART RX).
#define P_IN1 5
#define P_IN2 21
#define P_IN3 18
#define P_IN4 17
#define MAGNET_PIN 19  // relay "1" for the electromagnet     [single relay]
#define A_ENDSTOP_PIN 16  // base limit switch -> GND (INPUT_PULLUP). Active (a8 / hard stop) = LOW.

// ================ EDIT: microphone (MAX4466 analog -> GPIO34 / ADC1_CH6) ======
// GPIO34 is INPUT-ONLY (fine for a mic) and on ADC1, so — unlike ADC2 — it never
// conflicts with anything else here. Pi-driven: on "LISTEN" we stream audio
// frames then send AUDIO_END. On-device silence detection ends the window early;
// Whisper on the Pi makes the final call on what (if anything) was said. Turn-
// based with the motors: we never record and move at the same time.
#define MIC_PIN 34
const int MIC_FRAME_SAMPLES = 256;                 // samples per streamed frame
const unsigned long MIC_RECORD_MS = 6000;          // hard cap: always stop by 6 s
const int MIC_SILENCE_THRESHOLD = 800;             // peak deviation below this = silence (ambient ~500, speech ~2047); tune to the room
const unsigned long MIC_START_TIMEOUT_MS = 2800;   // wait this long for speech to BEGIN, else stop ("heard nothing")
const unsigned long MIC_SILENCE_HOLD_MS = 2000;    // stop after this much trailing quiet once speech began

// ================ EDIT: magnet relay =========================================
const bool MAGNET_ACTIVE_LOW = true;       // driver is active-LOW after the rewire (LOW energizes).
                                           // Was false; flipped because the magnet was ON at boot and
                                           // 'MAG ON' turned it OFF. Now HIGH = off (boot-safe), LOW = on.
const unsigned long AXIS_STAGGER_MS = 60;  // gap between axes: only ONE motor runs at a time

// ================ EDIT: rail (R) stepper calibration =========================
// Cart is a 28BYJ-48 (4096 half-steps/rev) on the OLD DC gearbox -> expect a high
// steps/mm and a SLOW drive. steps/mm = half-steps to move the cart 1 mm.
// PLACEHOLDER — CALIBRATE: `JOG R<steps>`, measure the mm travelled, R_STEPS_PER_MM
// = steps / mm. (Or use scripts/tune.py, which tunes RSPM live over CAL.)
// steps/mm from the rail's step range: 0 (cart nearest the tower = home) to a1 at
// the far end. a1 is ~18000-20000 half-steps out; assume 18000 for now. So
// R_STEPS_PER_MM = 18000 / (R_MAX_MM - R_MIN_MM) = 18000 / (387 - 117.5) ~= 66.8.
// Tune it (or a1's step count) with JOG R / scripts/tune.py.
const float R_STEPS_PER_MM = 50.0f;  // SAFE UNDERSHOOT: 63.08 drove a1 PAST the far stop and slammed
                                     // it. 50 lands a1 ~2 squares short (no contact). CALIBRATE UP:
                                     // jog the cart out, note the step count where it JUST reaches a1,
                                     // keep a few-mm margin, R_STEPS_PER_MM = that_count / (387-117.5).
                                     // Never let a move drive INTO the hard stop (open-loop, unsensed).
const int R_STEP_DIR = -1;           // physical coil direction, applied in railStep so it governs EVERY
                                     // rail move (JOG/GOTO/MOVE/HOME): +1 = advancing the phase drives the
                                     // cart OUTWARD, -1 inverts it. The step COUNT convention is unchanged
                                     // (+ = outward, 0 = inner home). Was +1; the post-rebuild rewire
                                     // reversed the rail (+ drove inward), so now -1.
// HALF-step drive + MICROSECOND timing. The speed-up over the original delay(3 ms)
// comes from the shorter us period here (delay() can't do sub-ms; 0.1 truncated to
// 0 = too fast = stall). Lower = faster; raise if the loaded gearbox buzzes/stalls.
const unsigned long R_STEP_DELAY_US = 1500;  // microseconds per HALF step (was 3000)

// ================ EDIT: base (A) stepper calibration =========================
// steps/deg = motor half-steps per rev (4096) * gear ratio / 360. Fine-tune from a
// corner with scripts/tune.py or `JOG A<steps>`: new = A_STEPS_PER_DEG * (cmd/achieved).
const float A_STEPS_PER_DEG = 43.284f;    // ~3.8x the 1:1 value (11.38)
const int A_STEP_DIR = +1;                // step sign for +degrees; flip to invert
const unsigned long A_STEP_DELAY_MS = 4;  // ms per half-step. HIGHER = slower = MORE torque
// Gearbox BACKLASH: on every direction reversal the base loses ~this many motor
// steps to gear slack before the output moves. baseStep() adds them back on a
// reversal so the tracked position stays exact (see g_aDir). ~100 measured.
const long A_BACKLASH_STEPS = 100;
// -- base limit switch: absolute zero reference (a8 side + hard stop) ---------
// A normally-open switch from A_ENDSTOP_PIN to GND (INPUT_PULLUP -> pressed = LOW).
// It sits where the base is aligned to a8, with a HARD STOP just past it — the base
// may never rotate beyond. HOME rotates toward it, then derives the (unchanged)
// centerline zero from A_ENDSTOP_STEPS, so the base self-homes (no hand-placing) and
// open-loop drift can't accumulate across games. The rail + winch have no switch.
const bool A_ENDSTOP_ACTIVE_LOW = true;   // switch to GND + internal pull-up: pressed = LOW
const int  A_HOME_DIR = +1;               // step sign that rotates TOWARD the switch (+theta = a8
                                          // side; see winch_offset note). Flip if HOME runs AWAY.
const long A_HOME_MAX_STEPS = 6000;       // seek travel cap (~full sweep + margin) before giving up
const long A_HOME_BACKOFF_STEPS = 200;    // release + slow re-approach for a repeatable trigger edge
// OUTPUT step count from the centerline zero (0) out to the switch. a8 is ~+32 deg,
// so ~ +32 * A_STEPS_PER_DEG (~+1385). 0 DISABLES switch-homing (count-home fallback,
// as before). MEASURE it once: power on at centerline, HOME, then run `SEEK` (or the
// anchor/tune `seek` command) — it reports the number. Bake it here + reflash; also
// settable live via `CAL AEND <n>`.
const long A_ENDSTOP_STEPS = 956;           // 0 until measured (keeps the old count-home behavior)

// ================ EDIT: soft limits & homing =================================
// R is the CART's radial position from the pivot. The INNER stop is home (= r_min);
// boot MUST start with the cart there. a1 sits at the FAR hard stop (r_max).
const float R_MIN_MM = 117.5f;   // cart R at the INNER mechanical stop (= home = step 0)
const float R_MAX_MM = 387.0f;   // cart R at the FAR stop (= a1). Soft clamp = the physical stop.
const float A_MIN_DEG = -55.0f;  // reachable sweep (h1 side)
const float A_MAX_DEG = 55.0f;   // a8 side is ALSO bounded by the limit switch / hard stop
                                 // (the runtime guard stops any move that reaches it).
const float R_HOME_MM = 117.5f;   // PARK pose = cart fully in against the inner stop (= r_min)
const float A_HOME_DEG = 1.063f;  // pivot-frame angle of the PARK/boot bearing (base zero offset).
                                  // Re-zero to 0.0 if you re-align home by hand.

// ================ EDIT: winch (28BYJ-48 / ULN2003), TWO-POSITION =============
// Two working positions a fixed step count apart: TRAVEL (up, start/rest) and
// PICK (down). Always begins at TRAVEL and returns to TRAVEL after each move.
const long WINCH_STROKE_STEPS = 7500;         // half-steps between TRAVEL and PICK
const int P_UP_STEP_DIR = -1;                 // step sign that RAISES the magnet (flip to invert)
const unsigned long WINCH_STEP_DELAY_MS = 2;  // per half-step (speed)
const float WINCH_PICK_BELOW_MM = 30.0f;      // a PULLEY height below this = drop to PICK
const float P_MAX_HEIGHT_MM = 80.0f;          // nominal TRAVEL/parked height (STATUS + HOME)
// After a move swings the arm/cart, the magnet on its string keeps swinging. Dwell
// this long at the END of every move so it settles before the winch can lower.
const unsigned long WINCH_SETTLE_MS = 500;  // string-settle wait after each move

// 28BYJ-48 half-step sequence (IN1..IN4). Reverse = walk it backwards.
const uint8_t HALFSTEP[8][4] = {
  { 1, 0, 0, 0 },
  { 1, 1, 0, 0 },
  { 0, 1, 0, 0 },
  { 0, 1, 1, 0 },
  { 0, 0, 1, 0 },
  { 0, 0, 1, 1 },
  { 0, 0, 0, 1 },
  { 1, 0, 0, 1 },
};
// FULL-step sequence (TWO coils energized each state). Half the states per rev of
// half-stepping -> ~2x the travel per step AND more torque, at half the
// resolution. The rail uses this so its slow gearbox moves quicker.
const uint8_t FULLSTEP[4][4] = {
  { 1, 1, 0, 0 },
  { 0, 1, 1, 0 },
  { 0, 0, 1, 1 },
  { 1, 0, 0, 1 },
};

// ----------------------------- state -----------------------------------------
bool g_estopped = false;
bool g_magnetOn = false;

float g_curH = P_MAX_HEIGHT_MM;  // tracked winch height (mm); boot = parked at the top
bool g_winchAtPick = false;      // false = TRAVEL (up, start), true = PICK (down)
int g_stepPhase = 0;             // winch half-step phase index
long g_wStepCount = 0;           // NET winch half-steps from boot (travel/top = 0)

float g_curA = A_HOME_DEG;  // tracked base angle (deg) — STATUS only; g_aStepCount is the truth
int g_aStepPhase = 0;       // base half-step phase index
long g_aStepCount = 0;      // base OUTPUT position in half-steps (backlash-free; home = 0)
int g_aDir = 0;             // last physical base move direction (+1/-1; 0 = unknown at boot)

float g_curR = R_MIN_MM;  // tracked cart radius (mm) — STATUS only; g_rStepCount is the truth
int g_rStepPhase = 0;     // rail half-step phase index
long g_rStepCount = 0;    // NET physical half-steps from boot (inner home = 0)

// -- runtime-tunable calibration (CAL command; see scripts/tune.py) -----------
// Initialized from the #define defaults, overwritten live over serial so the
// calibrator can tune WITHOUT a reflash. Bake the finals back into the #defines.
float g_aStepsPerDeg = A_STEPS_PER_DEG;  // base steps/deg (scale)
float g_aHomeDeg = A_HOME_DEG;           // base home/zero offset (deg)
float g_rStepsPerMm = R_STEPS_PER_MM;    // rail steps/mm (scale)
long  g_aEndstopSteps = A_ENDSTOP_STEPS; // base switch offset from centerline 0 (0 = disabled)

char g_line[96];     // line currently being assembled
char g_argline[96];  // clean copy of the last full line, for arg parsing
uint8_t g_len = 0;

// ----------------------------- helpers ---------------------------------------
float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void magnetWrite(bool on) {
  digitalWrite(MAGNET_PIN, (on != MAGNET_ACTIVE_LOW) ? HIGH : LOW);
}

void replyOK(const char* extra = nullptr) {
  if (extra && *extra) {
    Serial.print("OK ");
    Serial.println(extra);
  } else {
    Serial.println("OK");
  }
}
void replyErr(const char* msg) {
  Serial.print("ERR ");
  Serial.println(msg);
}

// Find a token like "R12.34" in a space-delimited (mutable) string.
bool readKeyed(char* str, char key, float* out) {
  for (char* tok = strtok(str, " "); tok; tok = strtok(nullptr, " ")) {
    if (tok[0] == key && tok[1] != '\0') {
      *out = atof(tok + 1);
      return true;
    }
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

// ----------------------------- winch (stepper) -------------------------------
void winchWritePhase(int phase) {
  digitalWrite(P_IN1, HALFSTEP[phase][0]);
  digitalWrite(P_IN2, HALFSTEP[phase][1]);
  digitalWrite(P_IN3, HALFSTEP[phase][2]);
  digitalWrite(P_IN4, HALFSTEP[phase][3]);
}
void winchRelease() {  // de-energize all winch coils (ESTOP / boot)
  digitalWrite(P_IN1, LOW);
  digitalWrite(P_IN2, LOW);
  digitalWrite(P_IN3, LOW);
  digitalWrite(P_IN4, LOW);
}
// Move the winch `steps` (signed) with `delayMs` between half-steps. Blocking.
// Leaves the last phase ENERGIZED so it holds the hanging magnet. Tracks the net
// step count from boot-home (travel/top = 0) so the winch can be jogged + homed
// like the other axes and its height read back via STEPS.
void winchStep(long steps, unsigned long delayMs) {
  int dir = (steps >= 0) ? 1 : -1;
  long n = labs(steps);
  for (long i = 0; i < n; i++) {
    g_stepPhase = (g_stepPhase + dir + 8) & 7;
    winchWritePhase(g_stepPhase);
    delay(delayMs);
  }
  g_wStepCount += steps;
}
// Return the winch to travel/top (step 0) by count — robust after free jogging.
void winchHome() {
  winchStep(-g_wStepCount, WINCH_STEP_DELAY_MS);
  g_wStepCount = 0;
  g_winchAtPick = false;
  g_curH = P_MAX_HEIGHT_MM;
}

void doPulley(float hmm) {
  // Two-position winch: below WINCH_PICK_BELOW_MM = drop to PICK, else climb to
  // TRAVEL. Move the fixed stroke only when the position changes; +stroke raises.
  bool wantPick = (hmm < WINCH_PICK_BELOW_MM);
  if (wantPick != g_winchAtPick) {
    long steps = WINCH_STROKE_STEPS * P_UP_STEP_DIR * (wantPick ? -1 : +1);
    winchStep(steps, WINCH_STEP_DELAY_MS);
    g_winchAtPick = wantPick;
  }
  g_curH = hmm;  // track the requested height for STATUS
}

// ----------------------------- base (stepper) --------------------------------
void baseWritePhase(int phase) {
  digitalWrite(A_IN1, HALFSTEP[phase][0]);
  digitalWrite(A_IN2, HALFSTEP[phase][1]);
  digitalWrite(A_IN3, HALFSTEP[phase][2]);
  digitalWrite(A_IN4, HALFSTEP[phase][3]);
}
void baseRelease() {  // de-energize all base coils
  digitalWrite(A_IN1, LOW);
  digitalWrite(A_IN2, LOW);
  digitalWrite(A_IN3, LOW);
  digitalWrite(A_IN4, LOW);
}
// --- base limit switch (a8 side / hard stop) = absolute zero reference --------
bool baseSwitchPressed() {
  return digitalRead(A_ENDSTOP_PIN) == (A_ENDSTOP_ACTIVE_LOW ? LOW : HIGH);
}
// One raw half-step toward `dir` (+1/-1). No position/backlash bookkeeping — used
// only by the switch seek, which cares about the physical edge, not the count.
void baseStepOnce(int dir) {
  g_aStepPhase = (g_aStepPhase + dir + 8) & 7;
  baseWritePhase(g_aStepPhase);
  delay(A_STEP_DELAY_MS);
}
// Rotate toward the switch until it triggers, stopping the instant it does (never
// slams the hard stop). Two-pass: release if already on it, fast approach, back off,
// slow re-approach for a repeatable edge. Leaves the base AT the switch; returns
// false if it isn't found within A_HOME_MAX_STEPS (broken / miswired / wrong dir).
bool baseSeekSwitch() {
  long i;
  if (baseSwitchPressed()) {                                        // already on it: release first
    for (i = 0; baseSwitchPressed() && i < A_HOME_MAX_STEPS; i++) baseStepOnce(-A_HOME_DIR);
    for (i = 0; i < A_HOME_BACKOFF_STEPS; i++) baseStepOnce(-A_HOME_DIR);
  }
  for (i = 0; !baseSwitchPressed() && i < A_HOME_MAX_STEPS; i++) baseStepOnce(A_HOME_DIR);
  if (!baseSwitchPressed()) return false;                           // never reached it
  for (i = 0; i < A_HOME_BACKOFF_STEPS; i++) baseStepOnce(-A_HOME_DIR);            // back off
  for (i = 0; !baseSwitchPressed() && i < A_HOME_BACKOFF_STEPS * 4; i++) baseStepOnce(A_HOME_DIR);
  return baseSwitchPressed();
}
// Turn the base motor `motorSteps` half-steps (signed). Raw — no position/backlash
// bookkeeping; used by baseStep() which owns that. Returns false (move TRUNCATED)
// if it would drive PAST the switch: once the offset is calibrated, the a8 hard stop
// is never ground into. Only guards toward the switch, so leaving home is unimpeded.
bool baseStepMotor(long motorSteps) {
  int dir = (motorSteps >= 0) ? 1 : -1;
  long n = labs(motorSteps);
  bool guard = (g_aEndstopSteps != 0);   // only once the switch offset is calibrated
  for (long i = 0; i < n; i++) {
    if (guard && dir == A_HOME_DIR && baseSwitchPressed()) return false;  // at the hard limit
    g_aStepPhase = (g_aStepPhase + dir + 8) & 7;
    baseWritePhase(g_aStepPhase);
    delay(A_STEP_DELAY_MS);
  }
  return true;
}
// Move the base OUTPUT by `steps` half-steps (signed), compensating gearbox
// backlash: on a direction reversal the motor first turns A_BACKLASH_STEPS extra
// to take up gear slack (output stationary), THEN the requested steps move the
// output. So g_aStepCount tracks the true OUTPUT position regardless of how many
// times we reverse — no drift, no need to nudge with repeated small moves.
void baseStep(long steps) {
  if (steps == 0) return;
  int dir = (steps > 0) ? 1 : -1;
  long comp = (g_aDir != 0 && dir != g_aDir) ? A_BACKLASH_STEPS : 0;  // reversal -> re-engage gears
  if (!baseStepMotor(steps + dir * comp)) {   // hit the switch: we ARE at the a8 reference
    g_aStepCount = g_aEndstopSteps;           // snap the OUTPUT position to the known offset
    g_aDir = A_HOME_DIR;
    Serial.println("# base limit switch reached; position re-zeroed to the a8 reference");
    return;
  }
  g_aStepCount += steps;   // output advanced by exactly `steps` (comp didn't move it)
  g_aDir = dir;
}
// Move the base to absolute angle `adeg` by ABSOLUTE step count: step 0 (= home =
// boot) is board angle g_aHomeDeg; the base advances g_aStepsPerDeg per degree.
// Changing ASPD/AHOME only remaps angles<->steps; homing just returns to 0.
void baseMoveTo(float adeg) {
  adeg = clampf(adeg, A_MIN_DEG, A_MAX_DEG);
  long target = lroundf((adeg - g_aHomeDeg) * g_aStepsPerDeg) * A_STEP_DIR;
  baseStep(target - g_aStepCount);
  g_curA = adeg;
}
void baseHome() {
  // With the switch calibrated (A_ENDSTOP_STEPS != 0): rotate to it, adopt its known
  // OUTPUT offset, then drive back to the centerline zero (output 0) — an ABSOLUTE
  // home that survives power cycles + open-loop drift, no hand-placing. Without it:
  // fall back to the sensorless count-back-to-0 (which needs a hand-placed boot home).
  if (g_aEndstopSteps != 0 && baseSeekSwitch()) {
    g_aStepCount = g_aEndstopSteps;   // physically at the switch = this many steps out
    g_aDir = A_HOME_DIR;              // gears last engaged toward the switch
    baseStep(-g_aStepCount);          // return to the centerline zero (output 0)
  } else {
    if (g_aEndstopSteps != 0) Serial.println("# WARN base switch not found; count-homing instead");
    baseStep(-g_aStepCount);          // sensorless: undo every net step since boot -> 0
  }
  g_curA = g_aHomeDeg;                // step 0 IS board angle g_aHomeDeg
}

// ----------------------------- rail (stepper) --------------------------------
// Same model as the base: step-counted from boot-home. Step 0 = the INNER stop
// (= R_MIN_MM); the cart advances g_rStepsPerMm per mm outward. Leaves coils
// energized after a move so the cart holds its radius against back-drive.
void railWritePhase(int phase) {  // HALF-step, straight order — this is what ROTATED.
  // (Full-step vibrated: the loaded gearbox can't make a whole-step jump at speed
  // and loses sync. Half-step's finer increments move it. Speed comes from the
  // microsecond timing below, not from full-stepping.)
  digitalWrite(R_IN1, HALFSTEP[phase][0]);
  digitalWrite(R_IN2, HALFSTEP[phase][1]);
  digitalWrite(R_IN3, HALFSTEP[phase][2]);
  digitalWrite(R_IN4, HALFSTEP[phase][3]);
}
void railRelease() {  // de-energize all rail coils (ESTOP / boot)
  digitalWrite(R_IN1, LOW);
  digitalWrite(R_IN2, LOW);
  digitalWrite(R_IN3, LOW);
  digitalWrite(R_IN4, LOW);
}
void railStep(long steps) {
  int dir = (steps >= 0) ? 1 : -1;
  long n = labs(steps);
  const unsigned long stepMs = R_STEP_DELAY_US / 1000;   // whole-ms part of the period
  const unsigned int  stepUs = R_STEP_DELAY_US % 1000;   // sub-ms remainder
  for (long i = 0; i < n; i++) {
    g_rStepPhase = (g_rStepPhase + dir * R_STEP_DIR + 8) & 7;  // R_STEP_DIR inverts the physical direction
    railWritePhase(g_rStepPhase);
    // Split the period so delay() (>=1 ms) YIELDS to the RTOS each step. A pure
    // delayMicroseconds() busy-wait over thousands of steps starves the idle task
    // -> watchdog / the ESP stops responding mid-move. yield() covers sub-ms rates.
    if (stepMs) delay(stepMs);
    if (stepUs) delayMicroseconds(stepUs);
    if ((i & 0x3F) == 0) yield();
  }
  g_rStepCount += steps;  // ground-truth physical position (inner home = 0)
}
void railMoveTo(float rmm) {
  rmm = clampf(rmm, R_MIN_MM, R_MAX_MM);
  long target = lroundf((rmm - R_MIN_MM) * g_rStepsPerMm);  // + = outward; R_STEP_DIR is applied in railStep
  railStep(target - g_rStepCount);
  g_curR = rmm;
}
void railHome() {
  // PURE STEP COUNT — never seeks a physical stop (the stepper can't take being
  // slammed). Just walks the net steps back to 0 (cart nearest the tower). Boot
  // MUST start the cart at 0; the R_MIN/R_MAX soft clamp keeps moves off both ends.
  railStep(-g_rStepCount);
  g_rStepCount = 0;
  g_curR = R_MIN_MM;
}

// ----------------------------- homing ----------------------------------------
// The BASE homes against its limit switch (absolute, no hand-placing) once the
// switch offset is calibrated; the winch + cart are sensorless and count back to 0,
// so boot MUST start them at home (winch up, cart at the inner stop). Base first,
// winch to travel, then the cart in.
void doHome() {
  g_estopped = false;
  char buf[96];
  snprintf(buf, sizeof(buf), "# HOME start: from R%.2f A%.2f H%.2f", g_curR, g_curA, g_curH);
  Serial.println(buf);
  baseHome();                 // base stepper -> physical 0 (home bearing)
  winchHome();                // winch stepper -> up/travel (its 0), by count
  baseRelease();              // free base coils (only one motor drawing at a time)
  delay(AXIS_STAGGER_MS);
  railHome();  // rail stepper -> inner home (count 0)
  snprintf(buf, sizeof(buf), "# HOME done: R%.2f A%.2f H%.2f", g_curR, g_curA, g_curH);
  Serial.println(buf);
}

// ----------------------------- motion ----------------------------------------
// Move to absolute (rmm, adeg) ONE AXIS AT A TIME. Rotate the base, RELEASE its
// coils (its gearbox holds the angle — a radial cart move exerts no torque about
// the base axis), then drive the cart. `feed` is ignored. A settle dwell at the
// end lets the magnet stop swinging before the winch lowers.
void doMoveRA(float rmm, float adeg) {
  baseMoveTo(adeg);  // rotating base: exact angle
  baseRelease();     // base holds via gearbox; frees the bus for the rail
  delay(AXIS_STAGGER_MS);
  railMoveTo(rmm);         // radial cart: exact mm (leaves coils energized to hold)
  delay(WINCH_SETTLE_MS);  // let the hanging magnet/string stop swinging
}

void doStatus() {
  char buf[80];
  snprintf(buf, sizeof(buf), "R%.2f A%.2f H%.2f MAG%d ENDR%d ENDA%d",
           g_curR, g_curA, g_curH, g_magnetOn ? 1 : 0, 0, baseSwitchPressed() ? 1 : 0);  // ENDA = base switch
  replyOK(buf);
}

void doEstop() {
  g_estopped = true;
  railRelease();
  baseRelease();
  winchRelease();
  magnetWrite(false);
  g_magnetOn = false;
  replyOK("ESTOP");
}

// Runtime calibration set/report: "CAL [ASPD v] [AHOME v] [RSPM v] [AEND v]". No
// args just reports. Lets scripts/tune.py tune the base scale/offset + rail steps/mm
// live, without a reflash. ASPD = base steps/deg, AHOME = base home offset (deg),
// RSPM = rail steps/mm, AEND = base limit-switch offset in steps (0 disables switch
// homing). RAM only — bake into the #defines to keep across a reboot.
void doCal() {
  char tmp[96];
  strncpy(tmp, g_argline, sizeof(tmp));
  tmp[sizeof(tmp) - 1] = '\0';
  strtok(tmp, " ");  // consume the "CAL" token
  for (char* key = strtok(nullptr, " "); key; key = strtok(nullptr, " ")) {
    char* val = strtok(nullptr, " ");
    if (!val) break;
    float v = atof(val);
    if (!strcmp(key, "ASPD")) g_aStepsPerDeg = v;
    else if (!strcmp(key, "AHOME")) g_aHomeDeg = v;
    else if (!strcmp(key, "RSPM")) g_rStepsPerMm = v;
    else if (!strcmp(key, "AEND")) g_aEndstopSteps = (long)v;  // base switch offset (0 disables)
  }
  char buf[96];
  snprintf(buf, sizeof(buf), "CAL ASPD%.4f AHOME%.4f RSPM%.4f AEND%ld",
           g_aStepsPerDeg, g_aHomeDeg, g_rStepsPerMm, g_aEndstopSteps);
  replyOK(buf);
}

// Bench-measure the base limit-switch offset. Requires a TRUSTED zero first (power
// on with the base at the centerline home, then HOME). Steps toward the switch,
// tracking the OUTPUT count, so g_aStepCount at the trigger = A_ENDSTOP_STEPS. Drops
// the guard during the seek, live-enables switch homing on success, and returns to
// the start either way. Bake the reported number into A_ENDSTOP_STEPS to persist.
void doSeekSwitch() {
  long start = g_aStepCount;
  long saved = g_aEndstopSteps;
  g_aEndstopSteps = 0;                 // drop the guard so we can drive onto the switch
  long i;
  for (i = 0; !baseSwitchPressed() && i < A_HOME_MAX_STEPS; i++) baseStep((long)A_HOME_DIR);
  bool found = baseSwitchPressed();
  long off = g_aStepCount - start;     // OUTPUT steps from the (trusted) zero to the switch
  g_aEndstopSteps = found ? off : saved;
  baseStep(start - g_aStepCount);      // return to where we started
  if (!found) { replyErr("SEEK: switch not found (check wiring / A_HOME_DIR sign)"); return; }
  char note[100];
  snprintf(note, sizeof(note),
           "# SEEK: base switch at %ld steps; bake A_ENDSTOP_STEPS=%ld & reflash to persist", off, off);
  Serial.println(note);
  char buf[48];
  snprintf(buf, sizeof(buf), "SEEK AEND%ld", off);
  replyOK(buf);
}

// --------------------------- command dispatch --------------------------------
void handleLine(char* line) {
  char* cmd = strtok(line, " ");  // g_argline still holds the full clean line
  if (!cmd) return;

  if (!strcmp(cmd, "PING")) {
    replyOK("PONG");
  } else if (!strcmp(cmd, "HOME")) {
    doHome();
    replyOK("HOMED");
  } else if (!strcmp(cmd, "STATUS")) {
    doStatus();
  } else if (!strcmp(cmd, "STEPS")) {
    // Raw physical step counts from boot-home (0). The authoritative position for
    // step-space calibration (scripts/anchor.py) — no mm/deg model involved.
    char buf[64];
    snprintf(buf, sizeof(buf), "STEPS A%ld R%ld W%ld", g_aStepCount, g_rStepCount, g_wStepCount);
    replyOK(buf);
  } else if (!strcmp(cmd, "ESTOP")) {
    doEstop();
  } else if (!strcmp(cmd, "CAL")) {
    doCal();  // set/report live calibration (allowed even when estopped)
  } else if (g_estopped) {
    replyErr("estopped; send HOME to clear");
  } else if (!strcmp(cmd, "MOVE")) {
    float r = g_curR;
    float a = g_curA;
    argKeyed('R', &r);
    argKeyed('A', &a);  // F is accepted but ignored
    doMoveRA(r, a);
    replyOK();
  } else if (!strcmp(cmd, "GOTO")) {
    // ABSOLUTE step-count move (the stepmap backend's primitive): drive the given
    // axes to an absolute half-step count (boot-home = 0), one at a time — base,
    // rail, then winch — matching the STEPS/anchor step space exactly. Base goes
    // through backlash comp + the switch guard; its coils release after (the gearbox
    // holds the angle) so the rail gets the full supply. No settle here: the caller
    // (lower/raise via a separate GOTO W) controls dwell.
    float v;
    bool headMoved = false;
    if (argKeyed('A', &v)) { baseStep((long)v - g_aStepCount); baseRelease(); headMoved = true; }
    if (argKeyed('R', &v)) { if (headMoved) delay(AXIS_STAGGER_MS); railStep((long)v - g_rStepCount); headMoved = true; }
    if (argKeyed('W', &v)) { if (headMoved) delay(AXIS_STAGGER_MS); winchStep((long)v - g_wStepCount, WINCH_STEP_DELAY_MS); }
    if (headMoved) delay(WINCH_SETTLE_MS);  // let the hanging magnet stop swinging before a lower
    replyOK();
  } else if (!strcmp(cmd, "JOG")) {
    // Bench calibration, RAW stepper jog: "JOG R<steps>" (rail), "JOG A<steps>"
    // (base), "JOG W<steps>" (winch). Signed half-steps, applied WITHOUT the
    // STEP_DIR so the host works in exact step-count space: JOG A<n> makes
    // g_aStepCount += n exactly (STEPS reads it back). Base jog goes through the
    // backlash compensation. Updates the counters (HOME re-zeros) but NOT mm/deg.
    float v;
    if (argKeyed('R', &v)) railStep((long)v);
    if (argKeyed('A', &v)) baseStep((long)v);
    if (argKeyed('W', &v)) winchStep((long)v, WINCH_STEP_DELAY_MS);
    replyOK();
  } else if (!strcmp(cmd, "SEEK")) {
    doSeekSwitch();  // base: find the limit switch, report/measure A_ENDSTOP_STEPS
  } else if (!strcmp(cmd, "PULLEY")) {
    float h = g_curH;
    argKeyed('H', &h);  // F ignored
    doPulley(h);
    replyOK();
  } else if (!strcmp(cmd, "MAG")) {
    char* arg = strtok(nullptr, " ");
    if (arg && !strcmp(arg, "ON")) {
      magnetWrite(true);
      g_magnetOn = true;
      replyOK();
    } else if (arg && !strcmp(arg, "OFF")) {
      magnetWrite(false);
      g_magnetOn = false;
      replyOK();
    } else replyErr("MAG expects ON or OFF");
  } else if (!strcmp(cmd, "LISTEN")) {
    recordWindow();   // streams audio frames, ends with AUDIO_END (no OK line)
  } else {
    replyErr("unknown command");
  }
}

// ------------------------------- setup/loop ----------------------------------
const char* resetReasonStr() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "POWERON";
    case ESP_RST_EXT: return "EXT";
    case ESP_RST_SW: return "SW";
    case ESP_RST_PANIC: return "PANIC/crash";
    case ESP_RST_INT_WDT: return "INT_WDT";
    case ESP_RST_TASK_WDT: return "TASK_WDT";
    case ESP_RST_WDT: return "WDT";
    case ESP_RST_BROWNOUT: return "BROWNOUT";
    case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
    default: return "UNKNOWN";
  }
}

// Record and stream audio frames to the Pi, then AUDIO_END. Stops early when no
// speech begins within MIC_START_TIMEOUT_MS, or after MIC_SILENCE_HOLD_MS of
// trailing quiet once speech began; always stops by MIC_RECORD_MS. Blocking by
// design: the Pi sends LISTEN and waits synchronously (no move is in flight).
// Frame = magic(0xAA 0x55) + length_LE(2) + length bytes of int16 PCM (~8 kHz).
void recordWindow() {
  unsigned long start = millis();
  bool speechStarted = false;
  unsigned long lastVoiceMs = start;
  while (millis() - start < MIC_RECORD_MS) {
    int16_t buf[MIC_FRAME_SAMPLES];
    int peak = 0;
    for (int i = 0; i < MIC_FRAME_SAMPLES; i++) {
      int v = analogRead(MIC_PIN) - 2048;    // 12-bit ADC centered
      buf[i] = (int16_t)(v << 4);            // 12-bit signed -> 16-bit
      if (abs(v) > peak) peak = abs(v);
      delayMicroseconds(100);               // ~8 kHz sample spacing
    }
    uint16_t byteLen = (uint16_t)(MIC_FRAME_SAMPLES * 2);
    Serial.write(0xAA); Serial.write(0x55);
    Serial.write((uint8_t)(byteLen & 0xFF));
    Serial.write((uint8_t)(byteLen >> 8));
    Serial.write((const uint8_t*)buf, byteLen);

    unsigned long now = millis();
    if (peak >= MIC_SILENCE_THRESHOLD) {
      speechStarted = true;
      lastVoiceMs = now;
    }
    if (!speechStarted) {
      if (now - start >= MIC_START_TIMEOUT_MS) break;      // nobody spoke
    } else if (now - lastVoiceMs >= MIC_SILENCE_HOLD_MS) {
      break;                                               // speaker finished
    }
  }
  Serial.println("AUDIO_END");
}

void setup() {
  Serial.begin(460800);   // 8 kHz * 16-bit audio + commands need > 115200 (see MAX4466 mic)

  // Magnet is ACTIVE-LOW: OFF = GPIO19 HIGH. Write HIGH BEFORE pinMode so the
  // internal pull-up holds it OFF during the pre-init window; then drive it. (A
  // brief float can still occur between chip reset and setup(); add an external
  // pull-UP to 3V3 on GPIO19 if you ever see the magnet tick on at power-up.)
  magnetWrite(false);
  pinMode(MAGNET_PIN, OUTPUT);
  magnetWrite(false);
  g_magnetOn = false;

  pinMode(R_IN1, OUTPUT);
  pinMode(R_IN2, OUTPUT);  // rail stepper
  pinMode(R_IN3, OUTPUT);
  pinMode(R_IN4, OUTPUT);
  pinMode(P_IN1, OUTPUT);
  pinMode(P_IN2, OUTPUT);  // winch stepper
  pinMode(P_IN3, OUTPUT);
  pinMode(P_IN4, OUTPUT);
  pinMode(A_IN1, OUTPUT);
  pinMode(A_IN2, OUTPUT);  // base stepper
  pinMode(A_IN3, OUTPUT);
  pinMode(A_IN4, OUTPUT);
  pinMode(A_ENDSTOP_PIN, INPUT_PULLUP);  // base limit switch to GND (a8 / hard stop)
  pinMode(MIC_PIN, INPUT);               // MAX4466 analog mic (ADC1, input-only pin)

  railRelease();   // rail coils off
  winchRelease();  // winch coils off
  baseRelease();   // base coils off

  char buf[64];
  snprintf(buf, sizeof(buf), "# esp32_chess ready (reset: %s)", resetReasonStr());
  Serial.println(buf);
}

void loop() {
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
