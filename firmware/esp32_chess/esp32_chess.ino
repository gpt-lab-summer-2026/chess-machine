/*
 * esp32_chess — motor controller firmware for the voice-controlled chess board.
 *
 * Drives a polar crane: a ROTARY base (theta) + a RADIAL railcart along the arm
 * (r), plus a pulley winch and one electromagnet. It speaks the line protocol
 * the host (chessmachine.motion.serial_esp32) expects:
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
 * the radial axis (mm from the pivot) and the rotary axis (degrees in the pivot
 * frame). Every motion command BLOCKS until the move finishes, then replies OK,
 * so the host never has to track motor state. Lines starting with '#' are debug.
 *
 * Requires the AccelStepper library (Library Manager: "AccelStepper").
 * Board: any ESP32 dev module. EDIT the pin + mechanics section for your wiring.
 */
#include <AccelStepper.h>

// ======================= EDIT: pins ==========================================
#define R_STEP_PIN     26
#define R_DIR_PIN      16
#define A_STEP_PIN     25
#define A_DIR_PIN      27
#define P_STEP_PIN     33
#define P_DIR_PIN      32
#define MOTOR_ENABLE   5     // shared driver enable, active LOW
#define R_ENDSTOP_PIN  13    // INPUT_PULLUP, LOW = pressed (radial, inner end)
#define A_ENDSTOP_PIN  14    // rotary home switch
#define P_TOP_ENDSTOP  15    // pulley fully-retracted switch
#define MAGNET_PIN     23    // MOSFET gate / relay (HIGH = energized)

// ==================== EDIT: mechanics ========================================
const float R_STEPS_PER_MM  = 80.0;   // radial railcart: microstepping + belt pitch
const float A_STEPS_PER_DEG = 80.0;   // rotary base: microstepping * gear ratio / 360
const float P_STEPS_PER_MM  = 80.0;   // pulley winch
const bool  R_DIR_INVERT    = false;
const bool  A_DIR_INVERT    = false;
const bool  P_DIR_INVERT    = false;
const int   R_HOME_DIR      = -1;     // sign of travel toward the radial endstop (inner)
const int   A_HOME_DIR      = -1;     // sign of travel toward the rotary endstop
const float R_HOME_MM       = 80.0;   // radius at the radial endstop (= r_min)
const float A_HOME_DEG      = -60.0;  // pivot-frame angle at the rotary endstop
const float R_MIN_MM        = 80.0;   // soft limits: reachable annulus
const float R_MAX_MM        = 300.0;
const float A_MIN_DEG        = -55.0; // soft limits: reachable sweep
const float A_MAX_DEG        = 55.0;
const bool  P_HAS_TOP_ENDSTOP = true;
const float P_MAX_HEIGHT_MM   = 80.0; // magnet height when the top switch trips

const float HOME_BACKOFF_MM  = 3.0;
const float HOME_SPEED_MM_S   = 10.0;
const float ROTARY_SPEED_DEG_S = 40.0; // rotary slew (the host feed F is radial mm/min)
const float MAX_FEED_MM_MIN  = 6000.0;
const float DEFAULT_FEED_MM_MIN = 3000.0;
const float ACCEL_MM_S2      = 800.0;
// =============================================================================

AccelStepper rStep(AccelStepper::DRIVER, R_STEP_PIN, R_DIR_PIN);
AccelStepper aStep(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper pStep(AccelStepper::DRIVER, P_STEP_PIN, P_DIR_PIN);

bool    g_estopped = false;
bool    g_magnetOn = false;
char    g_line[96];      // line currently being assembled
char    g_argline[96];   // clean copy of the last full line, for argument parsing
uint8_t g_len = 0;

// ----------------------------- helpers ---------------------------------------
float feedToSps(float feed_mm_min, float steps_per_mm) {
  if (feed_mm_min <= 0)              feed_mm_min = DEFAULT_FEED_MM_MIN;
  if (feed_mm_min > MAX_FEED_MM_MIN) feed_mm_min = MAX_FEED_MM_MIN;
  return (feed_mm_min / 60.0f) * steps_per_mm;
}

inline bool endstopPressed(int pin) { return digitalRead(pin) == LOW; }
void motorsEnabled(bool on) { digitalWrite(MOTOR_ENABLE, on ? LOW : HIGH); }

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
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

void applyDefaultSpeeds() {
  rStep.setMaxSpeed(feedToSps(DEFAULT_FEED_MM_MIN, R_STEPS_PER_MM));
  aStep.setMaxSpeed(ROTARY_SPEED_DEG_S * A_STEPS_PER_DEG);
  pStep.setMaxSpeed(feedToSps(DEFAULT_FEED_MM_MIN, P_STEPS_PER_MM));
}

// ----------------------------- homing ----------------------------------------
// Seek the endstop, back off, and call that physical point `homeValue` (in the
// axis's own units * stepsPerUnit), so absolute moves work straight after.
void homeAxis(AccelStepper& s, int endstopPin, int homeDir, float stepsPerUnit,
              float homeValue) {
  float sps = HOME_SPEED_MM_S * stepsPerUnit;
  s.setMaxSpeed(sps);
  s.setSpeed(homeDir * sps);
  while (!endstopPressed(endstopPin)) s.runSpeed();   // seek the switch
  s.setCurrentPosition((long)(homeValue * stepsPerUnit));
  // back off a touch, into the reachable range
  s.moveTo((long)((homeValue - homeDir * HOME_BACKOFF_MM) * stepsPerUnit));
  while (s.distanceToGo() != 0) s.run();
}

void homePulley() {
  float sps = HOME_SPEED_MM_S * P_STEPS_PER_MM;
  pStep.setMaxSpeed(sps);
  if (P_HAS_TOP_ENDSTOP) {
    pStep.setSpeed(+sps);                              // wind up to the top switch
    while (!endstopPressed(P_TOP_ENDSTOP)) pStep.runSpeed();
  }
  pStep.setCurrentPosition((long)(P_MAX_HEIGHT_MM * P_STEPS_PER_MM));
}

void doHome() {
  g_estopped = false;
  motorsEnabled(true);
  homeAxis(rStep, R_ENDSTOP_PIN, R_HOME_DIR, R_STEPS_PER_MM, R_HOME_MM);
  homeAxis(aStep, A_ENDSTOP_PIN, A_HOME_DIR, A_STEPS_PER_DEG, A_HOME_DEG);
  homePulley();
  applyDefaultSpeeds();
}

// ----------------------------- motion ----------------------------------------
void doMoveRA(float rmm, float adeg, float feed) {
  rmm  = clampf(rmm, R_MIN_MM, R_MAX_MM);
  adeg = clampf(adeg, A_MIN_DEG, A_MAX_DEG);
  rStep.setMaxSpeed(feedToSps(feed, R_STEPS_PER_MM));
  aStep.setMaxSpeed(ROTARY_SPEED_DEG_S * A_STEPS_PER_DEG);
  rStep.moveTo((long)(rmm * R_STEPS_PER_MM));
  aStep.moveTo((long)(adeg * A_STEPS_PER_DEG));
  while (rStep.distanceToGo() != 0 || aStep.distanceToGo() != 0) {
    rStep.run();
    aStep.run();
  }
}

void doPulley(float hmm, float feed) {
  if (hmm < 0)               hmm = 0;
  if (hmm > P_MAX_HEIGHT_MM)  hmm = P_MAX_HEIGHT_MM;
  pStep.setMaxSpeed(feedToSps(feed, P_STEPS_PER_MM));
  pStep.moveTo((long)(hmm * P_STEPS_PER_MM));
  while (pStep.distanceToGo() != 0) pStep.run();
}

void doStatus() {
  float rmm  = rStep.currentPosition() / R_STEPS_PER_MM;
  float adeg = aStep.currentPosition() / A_STEPS_PER_DEG;
  float hmm  = pStep.currentPosition() / P_STEPS_PER_MM;
  char buf[80];
  snprintf(buf, sizeof(buf), "R%.2f A%.2f H%.2f MAG%d ENDR%d ENDA%d",
           rmm, adeg, hmm, g_magnetOn ? 1 : 0,
           endstopPressed(R_ENDSTOP_PIN) ? 1 : 0,
           endstopPressed(A_ENDSTOP_PIN) ? 1 : 0);
  replyOK(buf);
}

void doEstop() {
  g_estopped = true;
  motorsEnabled(false);
  digitalWrite(MAGNET_PIN, LOW);
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
    float r = rStep.currentPosition() / R_STEPS_PER_MM;
    float a = aStep.currentPosition() / A_STEPS_PER_DEG;
    float f = DEFAULT_FEED_MM_MIN;
    argKeyed('R', &r);
    argKeyed('A', &a);
    argKeyed('F', &f);
    doMoveRA(r, a, f);
    replyOK();
  } else if (!strcmp(cmd, "PULLEY")) {
    float h = pStep.currentPosition() / P_STEPS_PER_MM;
    float f = DEFAULT_FEED_MM_MIN;
    argKeyed('H', &h);
    argKeyed('F', &f);
    doPulley(h, f);
    replyOK();
  } else if (!strcmp(cmd, "MAG")) {
    char* arg = strtok(nullptr, " ");
    if (arg && !strcmp(arg, "ON"))       { digitalWrite(MAGNET_PIN, HIGH); g_magnetOn = true;  replyOK(); }
    else if (arg && !strcmp(arg, "OFF")) { digitalWrite(MAGNET_PIN, LOW);  g_magnetOn = false; replyOK(); }
    else replyErr("MAG expects ON or OFF");
  } else {
    replyErr("unknown command");
  }
}

// ------------------------------- setup/loop ----------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_ENABLE, OUTPUT);
  motorsEnabled(true);
  pinMode(MAGNET_PIN, OUTPUT);
  digitalWrite(MAGNET_PIN, LOW);
  pinMode(R_ENDSTOP_PIN, INPUT_PULLUP);
  pinMode(A_ENDSTOP_PIN, INPUT_PULLUP);
  if (P_HAS_TOP_ENDSTOP) pinMode(P_TOP_ENDSTOP, INPUT_PULLUP);

  rStep.setPinsInverted(R_DIR_INVERT, false, false);
  aStep.setPinsInverted(A_DIR_INVERT, false, false);
  pStep.setPinsInverted(P_DIR_INVERT, false, false);
  rStep.setAcceleration(ACCEL_MM_S2 * R_STEPS_PER_MM);
  aStep.setAcceleration(ACCEL_MM_S2 * A_STEPS_PER_DEG);
  pStep.setAcceleration(ACCEL_MM_S2 * P_STEPS_PER_MM);
  applyDefaultSpeeds();

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
