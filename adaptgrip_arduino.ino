#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN 150
#define SERVO_MAX 600
#define STEP_PIN 3
#define DIR_PIN  4
#define EN_PIN   5
#define FSR_LEFT  A6
#define FSR_RIGHT A7

bool systemEnabled = false;
long stepperPos    = 0;

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void moveServo(int ch, int angle) {
  angle = constrain(angle, 0, 180);
  pca.setPWM(ch, 0, angleToPulse(angle));
}

void allOff() {
  for (int i = 0; i < 6; i++) pca.setPWM(i, 0, 0);
  digitalWrite(EN_PIN, HIGH);
  Serial.println("All OFF");
}

void stepperMove(int steps, bool cw) {
  if (steps <= 0) return;
  digitalWrite(EN_PIN, LOW);
  digitalWrite(DIR_PIN, cw ? HIGH : LOW);
  delay(10);
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(1000);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(1000);
  }
  digitalWrite(EN_PIN, HIGH);
  if (cw) stepperPos += steps;
  else    stepperPos -= steps;
  Serial.print("STEPPER:OK | pos:");
  Serial.println(stepperPos);
}

void goHome() {
  // ── Calibrated rest positions ──
  moveServo(0,   0); delay(300);  // Shoulder R  rest=0
  moveServo(1, 180); delay(300);  // Shoulder L  rest=180
  moveServo(2,   0); delay(300);  // Elbow       rest=0
  moveServo(3,  96); delay(300);  // Wrist Pitch rest=96  ← FIXED
  moveServo(4,   0); delay(300);  // Wrist Roll  rest=0
  moveServo(5,   0); delay(300);  // Gripper     rest=0
  Serial.println("HOME:OK");
}

void stepperHome() {
  if (stepperPos == 0) {
    Serial.println("STEPPER:ALREADY HOME");
    return;
  } else if (stepperPos > 0) {
    stepperMove(abs(stepperPos), false);
  } else {
    stepperMove(abs(stepperPos), true);
  }
  stepperPos = 0;
  Serial.println("STEPPER:HOME OK");
}

void setup() {
  Serial.begin(9600);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  pinMode(EN_PIN,   OUTPUT);
  digitalWrite(EN_PIN, HIGH);

  Wire.begin();
  pca.begin();
  pca.setPWMFreq(50);
  delay(500);

  // ── ALL OFF on startup — nothing moves ──
  allOff();

  Serial.println("================================");
  Serial.println("AdaptGrip Arduino Controller");
  Serial.println("================================");
  Serial.println("Servos OFF — waiting for Python");
  Serial.println("Supply can be ON safely");
  Serial.println("Arm will NOT move until Python");
  Serial.println("script sends START command");
  Serial.println("================================");
  Serial.println("Calibration:");
  Serial.println("  CH0 ShoulderR  rest=0");
  Serial.println("  CH1 ShoulderL  rest=180");
  Serial.println("  CH2 Elbow      rest=0");
  Serial.println("  CH3 WristPitch rest=96");
  Serial.println("  CH4 WristRoll  rest=0");
  Serial.println("  CH5 Gripper    rest=0");
  Serial.println("================================");
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START") {
      systemEnabled = true;
      stepperPos    = 0;
      goHome();
      Serial.println("SYSTEM:ON");

    } else if (cmd == "STOP") {
      systemEnabled = false;
      allOff();
      Serial.println("SYSTEM:OFF");

    } else if (cmd == "FSR") {
      int l = analogRead(FSR_LEFT);
      int r = analogRead(FSR_RIGHT);
      Serial.print("FSR:");
      Serial.print(l);
      Serial.print(":");
      Serial.println(r);

    } else if (systemEnabled) {

      if (cmd == "HOME") {
        goHome();
        stepperPos = 0;

      } else if (cmd == "STEPPERHOME") {
        stepperHome();

      } else if (cmd.startsWith("STEPPER:CW:")) {
        int steps   = cmd.substring(11).toInt();
        long newPos = stepperPos + steps;
        if (newPos > 550) steps = 550 - stepperPos;
        if (steps > 0) stepperMove(steps, true);
        else Serial.println("STEPPER: CW limit reached");

      } else if (cmd.startsWith("STEPPER:CCW:")) {
        int steps   = cmd.substring(12).toInt();
        long newPos = stepperPos - steps;
        if (newPos < -300) steps = stepperPos + 300;
        if (steps > 0) stepperMove(steps, false);
        else Serial.println("STEPPER: CCW limit reached");

      } else if (cmd.indexOf(':') != -1) {
        int colonIdx = cmd.indexOf(':');
        String joint = cmd.substring(0, colonIdx);
        int angle    = cmd.substring(colonIdx + 1).toInt();

        if      (joint == "SHOULDER_R") moveServo(0, angle);
        else if (joint == "SHOULDER_L") moveServo(1, angle);
        else if (joint == "ELBOW")      moveServo(2, angle);
        else if (joint == "WRIST_P")    moveServo(3, angle);
        else if (joint == "WRIST_R")    moveServo(4, angle);
        else if (joint == "GRIPPER")    moveServo(5, angle);

        Serial.print("OK:");
        Serial.println(cmd);

      } else {
        Serial.println("Unknown command");
      }

    } else {
      Serial.println("System OFF — send START first");
    }
  }
}
