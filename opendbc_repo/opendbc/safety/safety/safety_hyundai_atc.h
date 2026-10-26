#pragma once

// Opt-in classic Hyundai diagnostic output. This hook limits host requests; it
// cannot send cleanup frames after a host crash, or prove physical lamp output.
// ScanDoc kia.dat SHA256:
// 857f99af96c62075e494a7860e642dd7fddc2ac87a28195d676d42387ca0e114
// Palisade LX2 ICU: 0x770/0x778, 1003, 2fbc15/16 03/00. No other UDS allowed.

static const uint16_t HYUNDAI_PARAM_CLASSIC_ATC = 1024U;
static const uint32_t HYUNDAI_ATC_INPUT_MAX_US = 200000U;
static const uint32_t HYUNDAI_ATC_ACK_MAX_US = 500000U;
static const uint32_t HYUNDAI_ATC_SESSION_MIN_US = 1000000U;
static const uint32_t HYUNDAI_ATC_ACQUIRE_MIN_US = 200000U;
static const uint32_t HYUNDAI_ATC_RELEASE_MIN_US = 10000U;

enum {
  HYUNDAI_ATC_NONE = 0,
  HYUNDAI_ATC_SESSION = 1,
  HYUNDAI_ATC_ACQUIRE = 2,
  HYUNDAI_ATC_RELEASE = 3,
};

typedef struct {
  bool enabled;
  bool input_seen;
  bool input_clear;
  uint32_t input_ts;
  bool session_confirmed;
  uint8_t released;
  int pending;
  uint8_t pending_side;
  uint32_t pending_ts;
  bool session_sent;
  uint32_t session_ts;
  bool acquire_sent;
  uint32_t acquire_ts;
  bool release_sent[2];
  uint32_t release_ts[2];
} HyundaiAtcState;

static HyundaiAtcState hyundai_atc;

static void hyundai_atc_init(bool enabled) {
  hyundai_atc = (HyundaiAtcState){0};
  hyundai_atc.enabled = enabled;
}

static bool hyundai_atc_packet(const CANPacket_t *packet) {
  return (GET_BUS(packet) == 0) && (GET_LEN(packet) == 8) && !packet->fd &&
         !packet->extended && !packet->returned && !packet->rejected;
}

static void hyundai_atc_invalidate(void) {
  hyundai_atc.session_confirmed = false;
  hyundai_atc.released = 0U;
  // Retain only a cleanup transaction, which is allowed while disengaged.
  if (hyundai_atc.pending != HYUNDAI_ATC_RELEASE) {
    hyundai_atc.pending = HYUNDAI_ATC_NONE;
  }
}

static bool hyundai_atc_live(uint32_t now) {
  bool live = controls_allowed && !relay_malfunction && hyundai_atc.input_seen && hyundai_atc.input_clear &&
              ((now - hyundai_atc.input_ts) <= HYUNDAI_ATC_INPUT_MAX_US);
  if (!live) {
    hyundai_atc_invalidate();
  }
  if ((hyundai_atc.pending != HYUNDAI_ATC_NONE) &&
      ((now - hyundai_atc.pending_ts) > HYUNDAI_ATC_ACK_MAX_US)) {
    hyundai_atc_invalidate();
    hyundai_atc.pending = HYUNDAI_ATC_NONE;
  }
  return live;
}

static void hyundai_atc_rx(const CANPacket_t *packet) {
  if (hyundai_atc.enabled) {
    uint32_t now = microsecond_timer_get();
    (void)hyundai_atc_live(now);  // A new CGW1 must not erase an intervening stale-input gap.
    int addr = GET_ADDR(packet);
    if ((addr == 0x541) && (GET_BUS(packet) == 0) && !packet->returned && !packet->rejected) {
      hyundai_atc.input_seen = hyundai_atc_packet(packet);
      hyundai_atc.input_clear = false;
      if (hyundai_atc.input_seen) {
        uint8_t hazard = (GET_BYTE(packet, 4) >> 1U) & 3U;
        uint8_t lamp_left = (GET_BYTE(packet, 2) >> 3U) & 3U;
        uint8_t lamp_right = (GET_BYTE(packet, 7) >> 6U) & 3U;
        hyundai_atc.input_clear = !GET_BIT(packet, 42U) && !GET_BIT(packet, 41U) &&
                                  (hazard == 0U) && (lamp_left <= 1U) && (lamp_right <= 1U);
        hyundai_atc.input_ts = now;
      }
    }
    bool live = hyundai_atc_live(now);
    if ((addr == 0x778) && hyundai_atc_packet(packet) && (hyundai_atc.pending != HYUNDAI_ATC_NONE)) {
      bool match = false;
      if (hyundai_atc.pending == HYUNDAI_ATC_SESSION) {
        match = live && ((GET_BYTE(packet, 0) == 2U) || (GET_BYTE(packet, 0) == 6U)) &&
                (GET_BYTE(packet, 1) == 0x50U) && (GET_BYTE(packet, 2) == 3U);
        hyundai_atc.session_confirmed = match;
      } else {
        uint8_t control = (hyundai_atc.pending == HYUNDAI_ATC_ACQUIRE) ? 3U : 0U;
        match = (GET_BYTE(packet, 0) == 4U) && (GET_BYTE(packet, 1) == 0x6FU) &&
                (GET_BYTE(packet, 2) == 0xBCU) && (GET_BYTE(packet, 3) == (0x15U + hyundai_atc.pending_side)) &&
                (GET_BYTE(packet, 4) == control);
        if (match && (hyundai_atc.pending == HYUNDAI_ATC_RELEASE)) {
          hyundai_atc.released |= 1U << hyundai_atc.pending_side;
        }
      }
      if (match) {
        hyundai_atc.pending = HYUNDAI_ATC_NONE;
      } else {
        // Negative, malformed or wrong-DID/control replies never grant permission.
        hyundai_atc_invalidate();
      }
    }
  }
}

static bool hyundai_atc_tx(const CANPacket_t *packet) {
  bool tx = false;
  if (hyundai_atc.enabled && !relay_malfunction && hyundai_atc_packet(packet)) {
    uint32_t now = microsecond_timer_get();
    bool live = hyundai_atc_live(now);
    bool session = (GET_BYTES(packet, 0, 4) == 0x00031002U) && (GET_BYTES(packet, 4, 4) == 0U);
    bool io_control = (GET_BYTE(packet, 0) == 4U) && (GET_BYTE(packet, 1) == 0x2FU) &&
                      (GET_BYTE(packet, 2) == 0xBCU) &&
                      ((GET_BYTE(packet, 3) == 0x15U) || (GET_BYTE(packet, 3) == 0x16U)) &&
                      (GET_BYTE(packet, 5) == 0U) && (GET_BYTE(packet, 6) == 0U) && (GET_BYTE(packet, 7) == 0U);
    if (session && live && (!hyundai_atc.session_sent || ((now - hyundai_atc.session_ts) >= HYUNDAI_ATC_SESSION_MIN_US))) {
      hyundai_atc.session_sent = true;
      hyundai_atc.session_ts = now;
      hyundai_atc.session_confirmed = false;
      hyundai_atc.released = 0U;
      hyundai_atc.pending = HYUNDAI_ATC_SESSION;
      tx = true;
    } else if (io_control) {
      uint8_t side = GET_BYTE(packet, 3) - 0x15U;
      if ((GET_BYTE(packet, 4) == 0U) &&
          (!hyundai_atc.release_sent[side] || ((now - hyundai_atc.release_ts[side]) >= HYUNDAI_ATC_RELEASE_MIN_US))) {
        hyundai_atc.release_sent[side] = true;
        hyundai_atc.release_ts[side] = now;
        hyundai_atc.released &= ~(1U << side);
        hyundai_atc.pending = HYUNDAI_ATC_RELEASE;
        hyundai_atc.pending_side = side;
        tx = true;
      } else if ((GET_BYTE(packet, 4) == 3U) && live && hyundai_atc.session_confirmed &&
                 (hyundai_atc.released == 3U) && (hyundai_atc.pending == HYUNDAI_ATC_NONE) &&
                 (!hyundai_atc.acquire_sent || ((now - hyundai_atc.acquire_ts) >= HYUNDAI_ATC_ACQUIRE_MIN_US))) {
        hyundai_atc.acquire_sent = true;
        hyundai_atc.acquire_ts = now;
        hyundai_atc.released = 0U;  // Both release ACKs are required again before any next ON.
        hyundai_atc.pending = HYUNDAI_ATC_ACQUIRE;
        hyundai_atc.pending_side = side;
        tx = true;
      } else {
      }
    } else {
    }
    if (tx) {
      hyundai_atc.pending_ts = now;
    }
  }
  return tx;
}
