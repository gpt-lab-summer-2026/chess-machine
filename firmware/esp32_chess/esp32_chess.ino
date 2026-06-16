/*
 * esp32_chess — motor controller firmware for the voice-controlled chess board.
 *
 * Drives three stepper axes (X, Z gantry + pulley winch) and one electromagnet,
 * and speaks the line protocol the host (chessmachine.motion.serial_esp32) expects:
 *
 *   PING                          -> OK PONG
 *   HOME                          -> OK HOMED
 *   MOVE X<mm> Z<mm> [F<mm/min>]  -> OK
 *   PULLEY H<mm> [F<mm/min>]      -> OK
 *   MAG ON|OFF                    -> OK
 *   STATUS                        -> OK X<f> Z<f> H<f> MAG<0|1> ENDX<0|1> ENDZ<0|1>
 *   ESTOP                         -> OK ESTOP
 *
 * Every motion command BLOCKS until the move finishes, then replies OK, so the
 * host never has to track motor state. Lines starting with '#' are debug logs.
 *
 * Requires the AccelStepper library (Library Manager: "AccelStepper").
 * Board: any ESP32 dev module. EDIT the pin + mechanics section for your wiring.
 */
#include <AccelStepper.h>

// ======================= EDIT: pins ==========================================
#define X_STEP_PIN     26
#define X_DIR_PIN      16
#define Z_STEP_PIN     25
#define Z_DIR_PIN      27
#define P_STEP_PIN     33
#define P_DIR_PIN      32
#define MOTOR_ENABLE   5     // shared driver enable, active LOW
#define X_ENDSTOP_PIN  13    // INPUT_PULLUP, LOW = pressed
#define Z_ENDSTOP_PIN  14
#define P_TOP_ENDSTOP  15    // pulley fully-retracted switch
#define MAGNET_PIN     23    // MOSFET gate / relay (HIGH = energized)

// ==================== EDIT: mechanics ========================================
const float X_STEPS_PER_MM = 80.0;   // depends on belt/leadscrew + microstepping
const float Z_STEPS_PER_MM = 80.0;
const float P_STEPS_PER_MM = 80.0;
const bool  X_DIR_INVERT   = false;
const bool  Z_DIR_INVERT   = false;
const bool  P_DIR_INVERT   = false;
const int   X_HOME_DIR     = -1;     // sign of travel toward the X endstop
const int   Z_HOME_DIR     = -1;
const bool  P_HAS_TOP_ENDSTOP = true;
const float P_MAX_HEIGHT_MM   = 80.0; // magnet height when the top switch trips

const float HOME_BACKOFF_MM = 3.0;
const float HOME_SPEED_MM_S  = 10.0;
const float MAX_FEED_MM_MIN  = 6000.0;
const float DEFAULT_FEED_MM_MIN = 3000.0;
const float ACCEL_MM_S2      = 800.0;
// =============================================================================

AccelStepper xStep(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper zStep(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);
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

void replyOK(const char* extra = nullptr) {
  if (extra && *extra) { Serial.print("OK "); Serial.println(extra); }
  else                 { Serial.println("OK"); }
}
void replyErr(const char* msg) { Serial.print("ERR "); Serial.println(msg); }

// Find a token like "X12.34" in a space-delimited (mutable) string.
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
  xStep.setMaxSpeed(feedToSps(DEFAULT_FEED_MM_MIN, X_STEPS_PER_MM));
  zStep.setMaxSpeed(feedToSps(DEFAULT_FEED_MM_MIN, Z_STEPS_PER_MM));
  pStep.setMaxSpeed(feedToSps(DEFAULT_FEED_MM_MIN, P_STEPS_PER_MM));
}

// ----------------------------- homing ----------------------------------------
void homeAxis(AccelStepper& s, int endstopPin, int homeDir, float stepsPerMm) {
  float sps = HOME_SPEED_MM_S * stepsPerMm;
  s.setMaxSpeed(sps);
  s.setSpeed(homeDir * sps);
  while (!endstopPressed(endstopPin)) s.runSpeed();   // seek the switch
  s.setCurrentPosition(0);
  s.moveTo((long)(-homeDir * HOME_BACKOFF_MM * stepsPerMm));  // back off, that's zero
  while (s.distanceToGo() != 0) s.run();
  s.setCurrentPosition(0);
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
  homeAxis(xStep, X_ENDSTOP_PIN, X_HOME_DIR, X_STEPS_PER_MM);
  homeAxis(zStep, Z_ENDSTOP_PIN, Z_HOME_DIR, Z_STEPS_PER_MM);
  homePulley();
  applyDefaultSpeeds();
}

// ----------------------------- motion ----------------------------------------
void doMoveXZ(float xmm, float zmm, float feed) {
  xStep.setMaxSpeed(feedToSps(feed, X_STEPS_PER_MM));
  zStep.setMaxSpeed(feedToSps(feed, Z_STEPS_PER_MM));
  xStep.moveTo((long)(xmm * X_STEPS_PER_MM));
  zStep.moveTo((long)(zmm * Z_STEPS_PER_MM));
  while (xStep.distanceToGo() != 0 || zStep.distanceToGo() != 0) {
    xStep.run();
    zStep.run();
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
  float xmm = xStep.currentPosition() / X_STEPS_PER_MM;
  float zmm = zStep.currentPosition() / Z_STEPS_PER_MM;
  float hmm = pStep.currentPosition() / P_STEPS_PER_MM;
  char buf[80];
  snprintf(buf, sizeof(buf), "X%.2f Z%.2f H%.2f MAG%d ENDX%d ENDZ%d",
           xmm, zmm, hmm, g_magnetOn ? 1 : 0,
           endstopPressed(X_ENDSTOP_PIN) ? 1 : 0,
           endstopPressed(Z_ENDSTOP_PIN) ? 1 : 0);
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
    float x = xStep.currentPosition() / X_STEPS_PER_MM;
    float z = zStep.currentPosition() / Z_STEPS_PER_MM;
    float f = DEFAULT_FEED_MM_MIN;
    argKeyed('X', &x);
    argKeyed('Z', &z);
    argKeyed('F', &f);
    doMoveXZ(x, z, f);
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
  pinMode(X_ENDSTOP_PIN, INPUT_PULLUP);
  pinMode(Z_ENDSTOP_PIN, INPUT_PULLUP);
  if (P_HAS_TOP_ENDSTOP) pinMode(P_TOP_ENDSTOP, INPUT_PULLUP);

  xStep.setPinsInverted(X_DIR_INVERT, false, false);
  zStep.setPinsInverted(Z_DIR_INVERT, false, false);
  pStep.setPinsInverted(P_DIR_INVERT, false, false);
  xStep.setAcceleration(ACCEL_MM_S2 * X_STEPS_PER_MM);
  zStep.setAcceleration(ACCEL_MM_S2 * Z_STEPS_PER_MM);
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
