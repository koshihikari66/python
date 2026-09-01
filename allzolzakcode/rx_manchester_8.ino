/*
  레이저 수신 - 맨체스터 디코더 (IEEE 802.3 방식)
  Arduino Uno R4 WiFi + PP-A435 레이저 수신 모듈

  설계 원칙 (비트 밀림 / 오버샘플링 방지):
  - 핀을 주기적으로 digitalRead()로 폴링(오버샘플링)하지 않는다.
    대신 신호가 실제로 바뀌는 순간(엣지)에만 인터럽트가 걸리게 하고,
    그 순간에만 micros()로 시간을 재는 "엣지 인터벌 디코딩" 방식을 쓴다.
  - 매 판정은 "이전 엣지로부터 지금까지 걸린 시간"만 보고 하기 때문에
    기준 시각을 누적하지 않는다. 즉 매 엣지마다 실제 신호에 맞춰
    다시 동기를 잡으므로, 오래 수신해도 오차가 쌓여 비트가 밀리는
    일이 없다.
  - ISR은 시간 비교와 비트 판정만 하고 끝낸다. 실제 Serial 출력처럼
    느릴 수 있는 작업은 링 버퍼에 담아 loop()에서 처리한다
    (ISR을 무겁게 만들면 다음 엣지를 놓쳐 비트가 밀릴 수 있기 때문).

  ※ BIT_PERIOD_US는 반드시 송신측(tx_manchester.py)의 BIT_PERIOD_US와
     동일해야 한다. 한쪽만 바꾸면 디코딩이 깨진다.

  노이즈 대응:
  - 한 전이 구간 안에서 노이즈로 여러 번 튀는(바운스) 경우, 너무 촘촘한
    간격(MIN_GLITCH_US 미만)의 엣지는 기준 시각을 건드리지 않고 무시한다.
    (근본 해결은 수신단 RC 필터/슈미트 트리거 같은 하드�웨어 필터.)
  - 일정 시간(NO_SIGNAL_TIMEOUT_US) 동안 엣지가 전혀 없으면 "신호 없음"으로
    보고 시리얼에 '-'를 출력하고 동기 상태를 리셋한다.
*/

const uint8_t       RX_PIN               = 2;     // 인터럽트 지원 핀으로 배선에 맞게 수정
const unsigned long BIT_PERIOD_US        = 1000;  // 송신측과 동일하게!
const unsigned long HALF_PERIOD_US       = BIT_PERIOD_US / 2;
const unsigned long TOLERANCE_US         = HALF_PERIOD_US / 3;   // 허용 오차(약 ±33%)
const unsigned long MIN_GLITCH_US        = 50;    // 이보다 촘촘한 엣지는 노이즈로 보고 무시
const unsigned long NO_SIGNAL_TIMEOUT_US = BIT_PERIOD_US * 3;    // 이만큼 엣지가 없으면 "신호 없음"

// BER이 50%보다 훨씬 높게(예: 85%+) 나오면 노이즈가 아니라 "신호가 반대로
// 읽히고 있다"는 신호다 (수신 모듈이 active-low이거나 트랜지스터 구동단에서
// 로직이 한 번 뒤집힌 경우 등). 그럴 땐 이 값만 true로 바꿔서 재확인해보자.
const bool RX_INVERT = true;

inline uint8_t readLevel() {
  uint8_t raw = digitalRead(RX_PIN);
  return RX_INVERT ? (uint8_t)!raw : raw;
}

// ── ISR <-> loop() 간 공유용 링 버퍼 ──────────────────────────
const uint8_t BUF_SIZE = 64;
volatile uint8_t  bitBuffer[BUF_SIZE];
volatile uint8_t  bufHead = 0;
volatile uint8_t  bufTail = 0;
volatile uint32_t overflowCount = 0;   // loop()가 못 따라올 때 카운트(디버그용)

// ── 디코더 상태 (ISR 내부에서만 사용) ─────────────────────────
volatile unsigned long lastEdgeTime   = 0;
volatile bool          haveLastEdge   = false;
volatile bool          midFlag        = false;  // 짧은 간격을 하나 이미 봤는지
volatile unsigned long lastActivityUs = 0;       // 가장 최근에 "어떤 엣지든" 들어온 시각(노이즈 포함)

// ── PRBS7 자기동기 체커 (BER 측정용) ──────────────────────────
// 별도 프레임 동기 없이, 수신된 마지막 7비트를 그대로 체커의 레지스터로
// 삼아 다음 비트를 예측한다. 레지스터는 항상 "실제 수신값"으로 갱신되므로
// 언제 연결해도 몇 비트 안에 자동으로 동기(lock)가 걸린다.
// (송신측 tx_manchester.py의 generate_prbs7()와 동일한 다항식: x^7+x^6+1)
uint8_t  prbsReg     = 0;
uint8_t  prbsWarmup  = 0;   // 0~7: 아직 레지스터를 채우는 중인지
uint32_t totalBits   = 0;
uint32_t errorBits   = 0;

// ── 끊김(무신호) 통계 ─────────────────────────────────────────
// BER(받은 비트 중 오류 비율)과는 별개로, "얼마나 자주/오래 못 받았는지"를
// 따로 기록한다. 끊겼다 복구된 직후에는 PRBS 체커도 다시 워밍업시킨다.
uint32_t gapCount   = 0;
uint32_t totalGapUs = 0;

// ── 비정상 연속 길이 감지 ─────────────────────────────────────
// PRBS7은 이론상 1이 최대 7개, 0이 최대 6개까지만 연속될 수 있다.
// 그보다 길게 연속되면(특히 0 연속은 체커가 못 잡는 맹점이 있어서) 여기서
// 별도로 카운트한다.
const uint8_t MAX_VALID_RUN = 8;   // 여유를 둔 임계값(이론상 최대 7보다 크게)
uint8_t  runValue        = 2;      // 2 = 아직 시작 안 함(0/1이 아닌 값으로 초기화)
uint16_t runLength       = 0;
uint32_t abnormalRunCount = 0;

void trackRunLength(uint8_t bit) {
  if (bit == runValue) {
    runLength++;
  } else {
    runValue = bit;
    runLength = 1;
  }

  // 임계값(8비트)을 넘어선 순간, 그리고 그 이후로도 8비트마다 한 번씩
  // (계속 고착돼 있다면 반복 재시도) 재동기화를 시도한다.
  if (runLength >= MAX_VALID_RUN && (runLength % MAX_VALID_RUN) == 0) {
    abnormalRunCount++;

    prbsWarmup = 0;   // 레지스터가 퇴화된(같은 값 반복) 패턴으로 오염됨 -> 새로 워밍업

    noInterrupts();
    haveLastEdge = false;  // 엣지 판정 위상도 다음 엣지부터 새로 잡게 함
    midFlag = false;
    interrupts();

    Serial.print("\n[경고] 비정상 연속 감지(");
    Serial.print(runLength);
    Serial.println("비트, 값 동일) - 재동기화 시도");
  }
}

// ── 디버그 토글 (시리얼 명령: 'r' = 원시 간격 로그, 'b' = 비트 출력) ──
volatile bool rawLogEnabled  = false;
volatile bool bitPrintEnabled = true;

// ISR에서 loop()로 원시 엣지 정보를 넘기기 위한 링 버퍼
struct EdgeEvent {
  uint32_t delta;
  uint8_t  rawLevel;
  char     tag;   // 'G'=글리치무시 'S'=경계(비트아님) 's'=경계뒤데이터 'L'=긴간격데이터 'X'=동기이탈
};
const uint8_t EDGE_BUF_SIZE = 64;
volatile EdgeEvent edgeBuffer[EDGE_BUF_SIZE];
volatile uint8_t edgeHead = 0;
volatile uint8_t edgeTail = 0;

void pushEdgeEvent(uint32_t delta, uint8_t rawLevel, char tag) {
  if (!rawLogEnabled) return;  // 꺼져 있으면 오버헤드 없이 바로 리턴
  uint8_t next = (uint8_t)((edgeHead + 1) % EDGE_BUF_SIZE);
  if (next != edgeTail) {
    edgeBuffer[edgeHead].delta    = delta;
    edgeBuffer[edgeHead].rawLevel = rawLevel;
    edgeBuffer[edgeHead].tag      = tag;
    edgeHead = next;
  }
  // 꽉 차면 조용히 드롭 (디버그 모드이므로 일부 유실은 허용)
}

void checkPrbsBit(uint8_t rxBit) {
  if (prbsWarmup < 7) {
    // 초기 7비트는 동기 확보용으로만 채우고 에러 카운트는 하지 않는다
    prbsReg = (uint8_t)((prbsReg << 1) | rxBit) & 0x7F;
    prbsWarmup++;
    return;
  }

  uint8_t predicted = (uint8_t)(((prbsReg >> 6) ^ (prbsReg >> 5)) & 1);
  totalBits++;
  if (predicted != rxBit) {
    errorBits++;
  }
  // 예측값이 아니라 "실제 수신값"으로 갱신 -> 지속적으로 재동기화됨
  prbsReg = (uint8_t)((prbsReg << 1) | rxBit) & 0x7F;
}

void pushBit(uint8_t bit) {
  uint8_t next = (uint8_t)((bufHead + 1) % BUF_SIZE);
  if (next != bufTail) {
    bitBuffer[bufHead] = bit;
    bufHead = next;
  } else {
    overflowCount++;  // 버퍼가 가득 참 (정상 동작에서는 거의 발생하지 않음)
  }
}

void onEdge() {
  unsigned long now = micros();
  lastActivityUs = now;  // 신호가 살아있다는 것 자체는 노이즈든 진짜든 항상 기록

  if (!haveLastEdge) {
    lastEdgeTime = now;
    haveLastEdge = true;
    midFlag = false;
    return;
  }

  unsigned long delta = now - lastEdgeTime;

  if (delta < MIN_GLITCH_US) {
    // 진짜 전이라기엔 너무 촘촘함 -> 노이즈로 인한 순간 진동(바운스)로 보고 무시.
    // lastEdgeTime을 갱신하지 않아 원래의 "진짜 전이 후보" 시각을 그대로 유지한다.
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'G');
    return;
  }

  lastEdgeTime = now;   // 여기서부터는 '진짜' 엣지로 인정 -> 기준 시각 갱신 (누적 오차 없음)

  bool isShort = (delta > HALF_PERIOD_US - TOLERANCE_US) &&
                 (delta < HALF_PERIOD_US + TOLERANCE_US);
  bool isLong  = (delta > BIT_PERIOD_US - TOLERANCE_US) &&
                 (delta < BIT_PERIOD_US + TOLERANCE_US);

  if (isLong) {
    // 긴 간격(약 1비트 구간) = 항상 "비트 중앙 전이" -> 무조건 데이터.
    // 위상이 어긋나 있었더라도 여기서 자동으로 재동기화된다.
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'L');
    pushBit(readLevel());  // 802.3: 전이 후 레벨이 곧 비트값
    midFlag = false;
  } else if (isShort) {
    if (midFlag) {
      // 두 번째 짧은 간격 = 경계 전이 다음의 데이터 전이 -> 비트 확정
      pushEdgeEvent(delta, digitalRead(RX_PIN), 's');
      pushBit(readLevel());
      midFlag = false;
    } else {
      // 첫 번째 짧은 간격 = 데이터 전이 다음의 경계 전이 -> 데이터 아님, 대기
      pushEdgeEvent(delta, digitalRead(RX_PIN), 'S');
      midFlag = true;
    }
  } else {
    // 예상 범위 밖 간격 -> 노이즈/동기 이탈로 보고 리셋.
    // 다음 엣지부터 새로 동기를 잡는다 (실제 신호 자체로 재동기화).
    pushEdgeEvent(delta, digitalRead(RX_PIN), 'X');
    midFlag = false;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RX_PIN, INPUT);  // 수신 모듈 출력이 반대로 보이면 INPUT_PULLUP + 로직 반전 고려
  attachInterrupt(digitalPinToInterrupt(RX_PIN), onEdge, CHANGE);
  Serial.println("맨체스터(IEEE 802.3) 수신 대기 중...");
  Serial.println("명령: 'r' = 원시 간격 로그 on/off, 'b' = 비트 출력 on/off");
}

void loop() {
  // ── 시리얼 명령 처리 (모드 토글) ─────────────────────────────
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      rawLogEnabled = !rawLogEnabled;
      Serial.print("\n[설정] 원시 간격 로그: ");
      Serial.println(rawLogEnabled ? "ON" : "OFF");
    } else if (c == 'b' || c == 'B') {
      bitPrintEnabled = !bitPrintEnabled;
      Serial.print("\n[설정] 비트 출력: ");
      Serial.println(bitPrintEnabled ? "ON" : "OFF");
    }
  }

  // ── 원시 간격 로그 드레인 (rawLogEnabled일 때만 쌓였을 것) ────
  while (edgeTail != edgeHead) {
    noInterrupts();
    uint32_t delta    = edgeBuffer[edgeTail].delta;
    uint8_t  rawLevel = edgeBuffer[edgeTail].rawLevel;
    char     tag      = edgeBuffer[edgeTail].tag;
    edgeTail = (uint8_t)((edgeTail + 1) % EDGE_BUF_SIZE);
    interrupts();
    Serial.print("RAW delta=");
    Serial.print(delta);
    Serial.print("us level=");
    Serial.print(rawLevel);
    Serial.print(" tag=");
    Serial.println(tag);
  }

  bool gotBit = false;

  // 버퍼에 쌓인 비트를 순서대로 처리 (ISR과 별개 타이밍)
  while (bufTail != bufHead) {
    noInterrupts();
    uint8_t bit = bitBuffer[bufTail];
    bufTail = (uint8_t)((bufTail + 1) % BUF_SIZE);
    interrupts();
    if (bitPrintEnabled) Serial.print(bit);
    checkPrbsBit(bit);
    trackRunLength(bit);
    gotBit = true;
  }

  // ── 신호 없음 감지 ('-' 출력) ─────────────────────────────
  static unsigned long lastDashTime = 0;
  static bool          inGap        = false;
  static unsigned long gapStartUs   = 0;
  unsigned long nowUs = micros();

  if (gotBit) {
    lastDashTime = nowUs;  // 방금 실제 비트가 나왔으니 무신호 타이머 리셋

    if (inGap) {
      // 끊김이 끝나고 신호가 복구된 첫 비트 - 갭 통계 마감 + 체커 재워밍업
      unsigned long gapDur = nowUs - gapStartUs;
      totalGapUs += gapDur;
      gapCount++;
      inGap = false;

      prbsWarmup = 0;  // 레지스터에 갭 이전 데이터가 남아있으니 새로 7비트 워밍업
                        // (안 하면 재획득 직후 몇 비트가 가짜 에러로 잡힘)

      Serial.print("\n[복구] ");
      Serial.print(gapDur / 1000.0, 1);
      Serial.println("ms 끊김 후 신호 재수신");
    }
  } else {
    noInterrupts();
    unsigned long lastAct = lastActivityUs;
    interrupts();

    if ((nowUs - lastAct > NO_SIGNAL_TIMEOUT_US) &&
        (nowUs - lastDashTime > NO_SIGNAL_TIMEOUT_US)) {
      Serial.print('-');
      lastDashTime = nowUs;

      if (!inGap) {
        inGap = true;
        gapStartUs = lastAct;  // 실제 마지막 활동 시각을 갭 시작점으로 삼음
      }

      noInterrupts();
      haveLastEdge = false;  // 신호가 다시 들어오면 새로 동기부터 시작
      midFlag = false;
      interrupts();
    }
  }

  static uint32_t lastOverflowPrinted = 0;
  if (overflowCount != lastOverflowPrinted) {
    Serial.print(" [경고: 버퍼 오버플로 ");
    Serial.print(overflowCount);
    Serial.println("회 - loop()가 처리 속도를 못 따라감]");
    lastOverflowPrinted = overflowCount;
  }

  // ── 주기적 BER 리포트 (1초마다) ────────────────────────────
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
      Serial.print((double)errorBits / (double)totalBits, 8);
      Serial.print(" | gaps=");
      Serial.print(gapCount);
      Serial.print(" gapTime=");
      Serial.print(totalGapUs / 1000.0, 1);
      Serial.print("ms | abnormalRuns=");
      Serial.println(abnormalRunCount);
    }
  }
}
