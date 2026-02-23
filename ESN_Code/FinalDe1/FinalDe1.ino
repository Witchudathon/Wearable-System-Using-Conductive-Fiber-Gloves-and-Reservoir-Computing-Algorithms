#include <Arduino.h>
#include <math.h>

/*
  Glove Sensor (ESP32, UPPER divider)
  - Rf = 1.5 kΩ ทุกนิ้ว
  - แสดงผลต่อ "นิ้ว" ให้ชัดขึ้นด้วย:
      * EMA smoothing
      * Hysteresis + Consecutive frames
      * Relative suppression (ตัดสัญญาณรวมของมือออกบางส่วน)
      * Hold-time ค้างสถานะ ON ชั่วครู่ให้ดูชัด
  Serial: 115200 baud (CSV)
*/

// ===== Pins (ESP32 ADC1 only) =====
const int PINS[] = {33, 32, 35, 34, 39};
const int N = sizeof(PINS) / sizeof(PINS[0]);

// ===== Fixed R per channel (Ω) — ใช้จริง: 1.5k ทุกนิ้ว =====
const float R_FIXEDS[N] = {1500, 1500, 1500, 1500, 1500};

// ===== ADC & timing =====
const int  V_IN_mV        = 3300;   // แนะนำวัดจริงเช่น 3310 แล้วใส่ให้ตรง
const int  AVG_N          = 64;     // oversampling ให้นิ่ง
const int  PRINT_EVERY_MS = 100;    // ความถี่การพิมพ์
const int  EPS_mV         = 3;      // กันหารศูนย์/ปลายช่วง

// ===== Baseline =====
const unsigned long BASELINE_WARM_MS = 2000;  // หน่วงเวลาจับ baseline ตอนนิ่ง

// ===== Orientation (UPPER) =====
#define ORIENT_UPPER_GLOBAL  true   // 3.3V → Rs → ADC → Rf → GND

// ===== % mode =====
// 0: % สูงเมื่อ Rs ลด, 1: % สูงเมื่อ Rs เพิ่ม, 2: |Δ|/baseline (แนะนำเริ่มทดลอง)
#define PCT_MODE 2

// ===== Multi-finger detection (เปิดปกติ หลายนิ้วได้) =====
#define SINGLE_ACTIVE_MODE false

// ฮิสทีรีซิสบน "score" หลัง suppression (ค่าตั้งสำหรับ Rf=1.5k)
const float ON_TH[N]  = { 8,  8,  8,  8, 10};  // เปิดเมื่อ >= นี้
const float OFF_TH[N] = { 5,  5,  5,  5,  7};  // ดับเมื่อ <= นี้

// ต้องติด/ดับติดต่อกันกี่เฟรม
const int ON_CONSEC  = 3;
const int OFF_CONSEC = 2;

// EMA smoothing
const float ALPHA = 0.50f;           // 0.2–0.5 (มากขึ้น=นิ่งขึ้น)

// Relative suppression เพื่อตัด common hand motion
#define USE_REL_SUPPRESS 1
const float REL_BETA = 0.75f;        // 0.55–0.75 (มากขึ้น=ตัดรวมแรงขึ้น เห็นนิ้วเด่นขึ้น)

// ค้างสถานะ ON อย่างน้อยกี่ ms ให้ดูชัด
const unsigned long HOLD_MS = 200;

// ===== Helpers =====
static inline int readmVStable(int pin, int n = AVG_N) {
  (void)analogReadMilliVolts(pin); delayMicroseconds(120);
  (void)analogReadMilliVolts(pin); delayMicroseconds(120);
  long s = 0;
  for (int k = 0; k < n; k++) {
    s += analogReadMilliVolts(pin);
    delayMicroseconds(80);
  }
  return (int)(s / n);
}

/*
  UPPER: 3.3V -> Rs -> ADC -> Rf -> GND
  V = Vin * Rf / (Rs + Rf)  ->  Rs = Rf * (Vin - V) / V
*/
static inline float rs_from_v(float v_mV, float r_fixed, bool orient_upper) {
  const float Vin = (float)V_IN_mV;
  const float V   = v_mV;
  if (orient_upper) {
    if (V <= EPS_mV)       return INFINITY;
    if (V >= Vin - EPS_mV) return 0.0f;
    return r_fixed * ((Vin - V) / V);
  } else {
    // LOWER (สำรอง)
    if (V <= EPS_mV)       return 0.0f;
    if (V >= Vin - EPS_mV) return INFINITY;
    return (V * r_fixed) / (Vin - V);
  }
}

// median ของ float 5 ตัว (เรียงเล็กน้อย)
static float median5(const float a[5]) {
  float b[5]; for (int i=0;i<5;i++) b[i]=a[i];
  for (int i=1;i<5;i++){ float key=b[i]; int j=i-1; while(j>=0 && b[j]>key){ b[j+1]=b[j]; j--; } b[j+1]=key; }
  return b[2];
}

// ===== State buffers =====
float base_ohm[N], base_mv[N];
double acc_ohm[N], acc_mv[N];
unsigned long acc_cnt[N];
bool  warmed = false;

float ema_pct[N];      // % หลัง EMA (ก่อน suppression)
float score[N];        // % หลัง suppression (ใช้ตัดสิน ON/OFF)
bool  finger_on[N];
int   on_streak[N], off_streak[N];
unsigned long hold_until[N];

void setup(){
  Serial.begin(115200);
  delay(300);
  analogSetWidth(12);
  // ช่วงแรงดันค่อนข้างต่ำ (หลัก 20–300 mV) เหมาะกับ 0 dB เพื่อความละเอียด
  for (int i=0;i<N;i++) analogSetPinAttenuation(PINS[i], ADC_0db);

  for (int i=0;i<N;i++){
    acc_ohm[i]=0; acc_mv[i]=0; acc_cnt[i]=0; base_ohm[i]=NAN; base_mv[i]=NAN;
    ema_pct[i]=0; score[i]=0; finger_on[i]=false; on_streak[i]=0; off_streak[i]=0; hold_until[i]=0;
  }

  Serial.println(F(
    "t_ms,"
    "Thumb,Index,Middle,Ring,Little,"                       // Rs (ohm)
    "Thumb_pct,Index_pct,Middle_pct,Ring_pct,Little_pct,"   // % raw
    "Thumb_pctEma,Index_pctEma,Middle_pctEma,Ring_pctEma,Little_pctEma," // % ema
    "Thumb_score,Index_score,Middle_score,Ring_score,Little_score,"       // % หลัง suppression
    "Thumb_on,Index_on,Middle_on,Ring_on,Little_on,"        // states
    "Thumb_mV,Index_mV,Middle_mV,Ring_mV,Little_mV"
  ));
}

void loop(){
  static unsigned long last=0;
  static unsigned long t0 = millis();
  unsigned long now=millis();
  if (now-last < (unsigned long)PRINT_EVERY_MS) return;
  last=now;

  float ohm[N], pct_raw[N];
  int   mvArr[N];

  for (int i=0;i<N;i++){
    int mv = readmVStable(PINS[i], AVG_N);
    mvArr[i] = mv;

    float r = rs_from_v((float)mv, R_FIXEDS[i], ORIENT_UPPER_GLOBAL);
    ohm[i] = r;

    // เก็บค่า baseline ระหว่าง warm
    if (!warmed && isfinite(r)) { acc_ohm[i]+=r; acc_mv[i]+=mv; acc_cnt[i]+=1; }

    // คำนวณ % (raw) เทียบ baseline
    float s = 0.0f;
    if (warmed && isfinite(r) && isfinite(base_ohm[i]) && base_ohm[i] > 1e-6f) {
      #if PCT_MODE == 0
        s = 100.0f * (base_ohm[i] - r) / base_ohm[i]; if (s < 0) s = 0;
      #elif PCT_MODE == 1
        s = 100.0f * (r - base_ohm[i]) / base_ohm[i]; if (s < 0) s = 0;
      #else
        s = 100.0f * fabsf(r - base_ohm[i]) / base_ohm[i];
      #endif
      if (s > 200) s = 200;
      pct_raw[i] = s;
    } else {
      pct_raw[i] = 0;
    }

    // EMA smoothing
    if (!warmed) ema_pct[i] = 0;
    else         ema_pct[i] = (ema_pct[i]==0)? pct_raw[i] : (ALPHA*pct_raw[i] + (1.0f-ALPHA)*ema_pct[i]);
  }

  // จับ baseline หลัง warm เสร็จ
  if (!warmed && (now - t0) >= BASELINE_WARM_MS) {
    for (int i=0;i<N;i++){
      if (acc_cnt[i] > 0) {
        base_ohm[i] = (float)(acc_ohm[i] / (double)acc_cnt[i]);
        base_mv[i]  = (float)(acc_mv[i]  / (double)acc_cnt[i]);
      } else {
        base_ohm[i] = ohm[i];
        base_mv[i]  = (float)mvArr[i];
      }
    }
    warmed = true;
  }

  // ===== Relative suppression: ลดสัญญาณรวมของมือให้ออกเด่นทีละนิ้ว =====
  float med = 0.0f;
  if (warmed) {
    float tmp[5]; for (int i=0;i<5;i++) tmp[i]=ema_pct[i];
    med = median5(tmp);
    for (int i=0;i<N;i++){
      #if USE_REL_SUPPRESS
        float v = ema_pct[i] - REL_BETA * med;
        if (v < 0) v = 0;
        if (v > 200) v = 200;
        score[i] = v;
      #else
        score[i] = ema_pct[i];
      #endif
    }
  } else {
    for (int i=0;i<N;i++) score[i]=0;
  }

  // ===== Hysteresis + Hold time (หลายช่องเปิดได้) =====
  if (warmed) {
    for (int i=0;i<N;i++){
      if (!finger_on[i]) {
        if (score[i] >= ON_TH[i]) {
          on_streak[i]++;
          if (on_streak[i] >= ON_CONSEC) {
            finger_on[i] = true;
            hold_until[i] = now + HOLD_MS;
            on_streak[i]=0; off_streak[i]=0;
          }
        } else on_streak[i]=0;
      } else {
        // ต้องรอ hold หมด + ต่ำกว่า OFF_TH ติดต่อกัน
        if (now >= hold_until[i] && score[i] <= OFF_TH[i]) {
          off_streak[i]++;
          if (off_streak[i] >= OFF_CONSEC) {
            finger_on[i] = false;
            off_streak[i]=0; on_streak[i]=0;
          }
        } else off_streak[i]=0;
      }
    }
  }

  // ===== พิมพ์ CSV =====
  Serial.print(now);
  // Rs (ohm)
  for (int i=0;i<N;i++){
    Serial.print(',');
    if (isinf(ohm[i])) Serial.print(F("inf"));
    else               Serial.print(ohm[i], 3);
  }
  // % raw
  for (int i=0;i<N;i++){ Serial.print(','); Serial.print(pct_raw[i], 2); }
  // % ema
  for (int i=0;i<N;i++){ Serial.print(','); Serial.print(ema_pct[i], 2); }
  // score (หลัง suppression)
  for (int i=0;i<N;i++){ Serial.print(','); Serial.print(score[i], 2); }
  // on/off
  for (int i=0;i<N;i++){ Serial.print(','); Serial.print(finger_on[i] ? 1 : 0); }
  // mV
  for (int i=0;i<N;i++){ Serial.print(','); Serial.print(mvArr[i]); }
  Serial.println();
}
