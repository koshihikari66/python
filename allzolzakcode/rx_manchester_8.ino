/*
  레이저 수신 - 맨체스터 디코더 (IEEE 802.3)
  Arduino UNO R4 WiFi + PP-A435

  명령:
  r = 원시 엣지 로그 ON/OFF
  b = 비트 출력 ON/OFF
*/

const uint8_t RX_PIN = 2;

const unsigned long BIT_PERIOD_US = 1000;
const unsigned long HALF_PERIOD_US = BIT_PERIOD_US / 2;
const unsigned long TOLERANCE_US = HALF_PERIOD_US / 3;
const unsigned long MIN_GLITCH_US = 50;
const unsigned long NO_SIGNAL_TIMEOUT_US = BIT_PERIOD_US * 3;

const bool RX_INVERT = true;

inline uint8_t readLevel() {
  uint8_t raw = digitalRead(RX_PIN);
  return RX_INVERT ? (uint8_t)!raw : raw;
}

// 비트 링 버퍼
const uint8_t BUF_SIZE = 64;
volatile uint8_t bitBuffer[BUF_SIZE];
volatile uint8_t bufHead = 0;
volatile uint8_t bufTail = 0;
volatile uint32_t overflowCount = 0;

// Manchester 디코더 상태
volatile unsigned long lastEdgeTime = 0;
volatile bool haveLastEdge = false;
volatile bool midFlag = false;

// PRBS7 BER 측정
uint8_t prbsReg = 0;
uint8_t prbsWarmup = 0;
uint32_t totalBits = 0;
uint32_t errorBits = 0;

// Gap 측정
uint32_t gapCount = 0;
uint32_t totalGapUs = 0;

unsigned long lastGoodBitUs = 0;
bool haveGoodBit = false;

bool inGap = false;
unsigned long gapStartUs = 0;
unsigned long lastDashTime = 0;

// 비정상 연속 비트 감지
const uint8_t MAX_VALID_RUN = 8;

uint8_t runValue = 2;
uint16_t runLength = 0;
uint32_t abnormalRunCount = 0;

// Debug
volatile bool rawLogEnabled = false;
volatile bool bitPrintEnabled = true;

struct EdgeEvent {
  uint32_t delta;
  uint8_t rawLevel;
  char tag;
};

const uint8_t EDGE_BUF_SIZE = 64;
volatile EdgeEvent edgeBuffer[EDGE_BUF_SIZE];
volatile uint8_t edgeHead = 0;
volatile uint8_t edgeTail = 0;


// ============================================================
// Raw edge event
// ============================================================

void pushEdgeEvent(uint32_t delta, uint8_t rawLevel, char tag) {
  if (!rawLogEnabled) return;

  uint8_t next = (uint8_t)((edgeHead + 1) % EDGE_BUF_SIZE);

  if (next != edgeTail) {
    edgeBuffer[edgeHead].delta = delta;
    edgeBuffer[edgeHead].rawLevel = rawLevel;
    edgeBuffer[edgeHead].tag = tag;
    edgeHead = next;
  }
}


// ============================================================
// Bit buffer
// ============================================================

void pushBit(uint8_t bit) {
  uint8_t next = (uint8_t)((bufHead + 1) % BUF_SIZE);

  if (next != bufTail) {
    bitBuffer[bufHead] = bit;
    bufHead = next;
  } else {
    overflowCount++;
  }
}


// ============================================================
// PRBS7 BER
// ============================================================

void checkPrbsBit(uint8_t rxBit) {
  if (prbsWarmup < 7) {
    prbsReg = (uint8_t)((prbsReg << 1) | rxBit) & 0x7F;
    prbsWarmup++;
    return;
  }

  uint8_t predicted =
    (uint8_t)(((prbsReg >> 6) ^ (prbsReg >> 5)) & 1);

  totalBits++;

  if (predicted != rxBit) {
    errorBits++;
  }

  prbsReg = (uint8_t)((prbsReg << 1) | rxBit) & 0x7F;
}


// ============================================================
// 비정상 연속 비트 감지
// 중요: 여기서는 재동기화를 하지 않고 카운트만 함
// ============================================================

void trackRunLength(uint8_t bit) {
  if (bit == runValue) {
    runLength++;
  } else {
    runValue = bit;
    runLength = 1;
  }

  if (runLength >= MAX_VALID_RUN &&
      (runLength % MAX_VALID_RUN) == 0) {

    abnormalRunCount++;

    Serial.print("\n[경고] 비정상 연속 감지(");
    Serial.print(runLength);
    Serial.println("비트, 값 동일)");
  }
}


// ============================================================
// Edge interrupt
// ============================================================

void onEdge() {
  unsigned long now = micros();

  if (!haveLastEdge) {
    lastEdgeTime = now;
    haveLastEdge = true;
    midFlag = false;
    return;
  }

  unsigned long delta = now - lastEdgeTime;

  // 지나치게 짧은 엣지는 노이즈로 무시
  if (delta < MIN_GLITCH_US) {
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'G');
    return;
  }

  lastEdgeTime = now;

  bool isShort =
    (delta > HALF_PERIOD_US - TOLERANCE_US) &&
    (delta < HALF_PERIOD_US + TOLERANCE_US);

  bool isLong =
    (delta > BIT_PERIOD_US - TOLERANCE_US) &&
    (delta < BIT_PERIOD_US + TOLERANCE_US);

  if (isLong) {
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'L');

    pushBit(readLevel());

    midFlag = false;
  }

  else if (isShort) {
    if (midFlag) {
      pushEdgeEvent(delta, digitalRead(RX_PIN), 's');

      pushBit(readLevel());

      midFlag = false;
    } else {
      pushEdgeEvent(delta, digitalRead(RX_PIN), 'S');

      midFlag = true;
    }
  }

  else {
    // 예상 범위를 벗어난 엣지
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'X');

    // short/long 위상만 다시 잡음
    midFlag = false;
  }
}


// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);

  pinMode(RX_PIN, INPUT);

  attachInterrupt(
    digitalPinToInterrupt(RX_PIN),
    onEdge,
    CHANGE
  );

  Serial.println("맨체스터(IEEE 802.3) 수신 대기 중...");
  Serial.println("명령: 'r' = 원시 간격 로그 on/off, 'b' = 비트 출력 on/off");
}


// ============================================================
// Main loop
// ============================================================

void loop() {

  // Serial 명령
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'r' || c == 'R') {
      rawLogEnabled = !rawLogEnabled;

      Serial.print("\n[설정] 원시 간격 로그: ");
      Serial.println(rawLogEnabled ? "ON" : "OFF");
    }

    else if (c == 'b' || c == 'B') {
      bitPrintEnabled = !bitPrintEnabled;

      Serial.print("\n[설정] 비트 출력: ");
      Serial.println(bitPrintEnabled ? "ON" : "OFF");
    }
  }


  // Raw edge 로그 출력
  while (edgeTail != edgeHead) {
    noInterrupts();

    uint32_t delta = edgeBuffer[edgeTail].delta;
    uint8_t rawLevel = edgeBuffer[edgeTail].rawLevel;
    char tag = edgeBuffer[edgeTail].tag;

    edgeTail = (uint8_t)((edgeTail + 1) % EDGE_BUF_SIZE);

    interrupts();

    Serial.print("RAW delta=");
    Serial.print(delta);

    Serial.print("us level=");
    Serial.print(rawLevel);

    Serial.print(" tag=");
    Serial.println(tag);
  }


  // 수신 비트 처리
  while (bufTail != bufHead) {
    noInterrupts();

    uint8_t bit = bitBuffer[bufTail];
    bufTail = (uint8_t)((bufTail + 1) % BUF_SIZE);

    interrupts();

    unsigned long bitNowUs = micros();


    // gap에서 복구
    if (inGap) {
      unsigned long gapDur = bitNowUs - gapStartUs;

      if (gapDur >= NO_SIGNAL_TIMEOUT_US) {
        gapCount++;
        totalGapUs += gapDur;

        Serial.print("\n[복구] ");
        Serial.print(gapDur / 1000.0, 1);
        Serial.println("ms 실제 통신 끊김 후 신호 재수신");

        // gap 이후 BER 체커만 다시 워밍업
        prbsWarmup = 0;
      }

      inGap = false;
    }


    // 정상 디코딩 비트 시각 갱신
    lastGoodBitUs = bitNowUs;
    haveGoodBit = true;


    // 비트 출력
    if (bitPrintEnabled) {
      Serial.print(bit);
    }


    checkPrbsBit(bit);
    trackRunLength(bit);
  }


  // ==========================================================
  // Gap 감지
  // ==========================================================

  unsigned long nowUs = micros();

  if (haveGoodBit) {
    if (!inGap &&
        (nowUs - lastGoodBitUs >= NO_SIGNAL_TIMEOUT_US)) {

      inGap = true;
      gapStartUs = lastGoodBitUs;

      Serial.print('-');
      lastDashTime = nowUs;

      // 실제 gap이 발생했으므로 다음 엣지부터 새로 위상 획득
      noInterrupts();
      haveLastEdge = false;
      midFlag = false;
      interrupts();
    }

    else if (inGap &&
             (nowUs - lastDashTime >= NO_SIGNAL_TIMEOUT_US)) {

      Serial.print('-');
      lastDashTime = nowUs;
    }
  }


  // ==========================================================
  // 버퍼 overflow
  // ==========================================================

  static uint32_t lastOverflowPrinted = 0;

  if (overflowCount != lastOverflowPrinted) {
    Serial.print("\n[경고] 버퍼 오버플로 ");
    Serial.print(overflowCount);
    Serial.println("회");

    lastOverflowPrinted = overflowCount;
  }


  // ==========================================================
  // BER 리포트
  // ==========================================================

  static unsigned long lastReportMs = 0;
  unsigned long nowMs = millis();

  if (nowMs - lastReportMs >= 1000) {
    lastReportMs = nowMs;

    if (totalBits > 0) {
      Serial.print("\n[BER] bits=");
      Serial.print(totalBits);

      Serial.print(" errors=");
      Serial.print(errorBits);

      Serial.print(" BER=");
      Serial.print(
        (double)errorBits / (double)totalBits,
        8
      );

      Serial.print(" | gaps=");
      Serial.print(gapCount);

      Serial.print(" gapTime=");
      Serial.print(totalGapUs / 1000.0, 1);

      Serial.print("ms | abnormalRuns=");
      Serial.println(abnormalRunCount);
    }
  }
}