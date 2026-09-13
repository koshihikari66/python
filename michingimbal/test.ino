#include <Servo.h>

Servo rollServo;
Servo pitchServo;

void setup() {
  Serial.begin(115200);

  rollServo.attach(9);
  pitchServo.attach(10);

  rollServo.writeMicroseconds(1500);
  pitchServo.writeMicroseconds(1500);

  delay(5000);
}

void loop() {
  // 양쪽 중앙
  rollServo.writeMicroseconds(1500);
  pitchServo.writeMicroseconds(1500);
  delay(3000);

  // ROLL 테스트
  rollServo.writeMicroseconds(1000);
  delay(3000);
  rollServo.writeMicroseconds(1500);
  delay(3000);
  rollServo.writeMicroseconds(2000);
  delay(3000);
  rollServo.writeMicroseconds(1500);
  delay(3000);

  // PITCH 테스트
  pitchServo.writeMicroseconds(1000);
  delay(3000);
  pitchServo.writeMicroseconds(1500);
  delay(3000);
  pitchServo.writeMicroseconds(2000);
  delay(3000);
  pitchServo.writeMicroseconds(1500);
  delay(3000);
}