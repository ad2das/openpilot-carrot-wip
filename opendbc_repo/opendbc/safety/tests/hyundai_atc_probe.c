// Exercise the actual safety dispatcher + Hyundai hooks, with the standard fake
// STM timer only. No transport or physical vehicle is involved.
#include <assert.h>
#include <string.h>
#include "fake_stm.h"
#ifdef HYUNDAI_ATC_PROBE_RELEASE
#undef ALLOW_DEBUG
#endif
#include "can.h"
#include "faults.h"
#include "safety.h"
#include "drivers/can_common.h"

bool safety_tx_buffered_for_fwd = false;
static int checks = 0;
#define CHECK(condition) do { checks++; assert(condition); } while (0)

static CANPacket_t packet(int addr, const uint8_t data[8]) {
  CANPacket_t p = {0};
  p.addr = addr;
  p.data_len_code = 8;
  memcpy(p.data, data, 8);
  return p;
}

static const uint8_t session_data[8] = {2, 0x10, 3, 0, 0, 0, 0, 0};
static const uint8_t session_ack[8] = {2, 0x50, 3, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa};

static CANPacket_t control(uint8_t side, uint8_t value) {
  uint8_t data[8] = {4, 0x2f, 0xbc, 0x15 + side, value, 0, 0, 0};
  return packet(0x770, data);
}

static bool tx(CANPacket_t p) {
  can_set_checksum(&p);
  return safety_tx_hook(&p);
}

static void rx(CANPacket_t p) {
  can_set_checksum(&p);
  CHECK(safety_rx_hook(&p));
}

static void clear_input(void) {
  const uint8_t clear[8] = {0};
  rx(packet(0x541, clear));
}

static void reset_mode(uint16_t mode, uint16_t param) {
  CHECK(set_safety_hooks(mode, param) == 0);
  timer.CNT = 0;
  controls_allowed = true;
  clear_input();
}

static void reset(void) {
  reset_mode(SAFETY_HYUNDAI, 1024U | 8U);
}

static void response(uint8_t side, uint8_t value) {
  uint8_t data[8] = {4, 0x6f, 0xbc, 0x15 + side, value, 0xaa, 0xaa, 0xaa};
  rx(packet(0x778, data));
}

static void release(uint8_t side) {
  CHECK(tx(control(side, 0)));
  response(side, 0);
}

static void ready(void) {
  reset();
  CHECK(tx(packet(0x770, session_data)));
  rx(packet(0x778, session_ack));
  release(0);
  release(1);
}

static void test_opt_in_and_modes(void) {
  const uint16_t params[] = {0, 4, 8, 12, 64, 512};
  for (unsigned int i = 0; i < sizeof(params) / sizeof(params[0]); i++) {
    reset_mode(SAFETY_HYUNDAI, params[i]);
    CHECK(!tx(packet(0x770, session_data)));
    CHECK(!tx(control(0, 0)));
    CHECK(!tx(control(1, 3)));
  }
  reset_mode(SAFETY_HYUNDAI_LEGACY, 1024U | 8U);
  CHECK(!tx(packet(0x770, session_data)));
  CHECK(!tx(control(0, 0)));
  reset_mode(SAFETY_HYUNDAI_CANFD, 1024U);
  CHECK(!tx(control(0, 0)));
  reset_mode(SAFETY_HYUNDAI, 1024U | 16U);
  CHECK(!tx(control(0, 0)));
  for (uint16_t config = 0; config <= 12; config += 4) {
    reset_mode(SAFETY_HYUNDAI, 1024U | config);
    CHECK(tx(packet(0x770, session_data)));
  }
}

static void test_exact_wire_shape(void) {
  const uint8_t bad[][8] = {
    {2, 0x3e, 0, 0, 0, 0, 0, 0}, {1, 0x20, 0, 0, 0, 0, 0, 0},
    {2, 0x10, 1, 0, 0, 0, 0, 0}, {4, 0x2f, 0xbc, 0x15, 2, 0, 0, 0},
    {4, 0x2f, 0xbc, 0, 2, 0, 0, 0}, {4, 0x2f, 0xbc, 0x17, 3, 0, 0, 0},
    {4, 0x2f, 0xbd, 0x15, 0, 0, 0, 0}, {0x10, 4, 0x2f, 0xbc, 0x15, 3, 0, 0},
  };
  for (unsigned int i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
    ready();
    CHECK(!tx(packet(0x770, bad[i])));
  }
  for (int dlc = 0; dlc < 16; dlc++) {
    if (dlc == 8) continue;
    reset();
    CANPacket_t p = control(0, 0);
    p.data_len_code = dlc;
    CHECK(!tx(p));
  }
  for (int bus = 1; bus < 8; bus++) {
    reset();
    CANPacket_t p = control(0, 0);
    p.bus = bus;
    CHECK(!tx(p));
  }
  for (int flag = 0; flag < 4; flag++) {
    reset();
    CANPacket_t p = control(0, 0);
    p.fd = flag == 0;
    p.extended = flag == 1;
    p.returned = flag == 2;
    p.rejected = flag == 3;
    CHECK(!tx(p));
  }
  for (int b = 3; b < 8; b++) {
    reset();
    CANPacket_t p = packet(0x770, session_data);
    p.data[b] = 1;
    CHECK(!tx(p));
  }
  for (int b = 5; b < 8; b++) {
    ready();
    CANPacket_t p = control(0, 3);
    p.data[b] = 1;
    CHECK(!tx(p));
    p = control(1, 0);
    p.data[b] = 0xaa;
    CHECK(!tx(p));
  }
}

static void test_input_gates_and_invalidation(void) {
  CHECK(set_safety_hooks(SAFETY_HYUNDAI, 1024U | 8U) == 0);
  timer.CNT = 0;
  controls_allowed = true;
  CHECK(!tx(packet(0x770, session_data)));
  clear_input();
  controls_allowed = false;
  CHECK(!tx(packet(0x770, session_data)));
  CHECK(tx(control(0, 0)));  // Cleanup never requires controls or a session.
  controls_allowed = true;
  timer.CNT = 200001;
  CHECK(!tx(packet(0x770, session_data)));
  const int bad_bits[] = {41, 42, 33, 34, 20, 63};
  for (unsigned int i = 0; i < sizeof(bad_bits) / sizeof(bad_bits[0]); i++) {
    ready();
    uint8_t data[8] = {0};
    data[bad_bits[i] / 8] = 1U << (bad_bits[i] % 8);
    rx(packet(0x541, data));
    CHECK(!tx(control(0, 3)));
    CHECK(!tx(packet(0x770, session_data)));
    clear_input();
    CHECK(!tx(control(0, 3)));  // Clear input alone cannot restore old permission.
    timer.CNT = 10000;
    CHECK(tx(control(0, 0)));
  }
  ready();
  timer.CNT = 200001;
  clear_input();  // Refresh after a gap must invalidate the earlier session.
  CHECK(!tx(control(0, 3)));
  ready();
  controls_allowed = false;
  clear_input();
  controls_allowed = true;
  clear_input();
  CHECK(!tx(control(0, 3)));
  ready();
  CANPacket_t malformed = packet(0x541, (uint8_t[8]){0});
  malformed.data_len_code = 7;
  rx(malformed);
  CHECK(!tx(control(0, 3)));
  ready();
  uint8_t lit[8] = {0, 0, 8, 0, 0, 0, 0, 64};
  rx(packet(0x541, lit));
  CHECK(tx(control(0, 3)));  // Valid lamp feedback is not a stalk input.
}

static void test_ack_sequence_and_echo_rejection(void) {
  reset();
  CHECK(tx(packet(0x770, session_data)));
  rx(packet(0x778, (uint8_t[8]){6, 0x50, 3, 0, 0x32, 1, 0xf4, 0xaa}));
  release(0);
  release(1);
  CHECK(tx(control(1, 3)));  // Session responses may contain ECU timing fields.
  reset();
  rx(packet(0x778, session_ack));
  response(0, 0);
  response(1, 0);
  CHECK(!tx(control(0, 3)));  // Unsolicited replies cannot manufacture ownership.
  CHECK(tx(packet(0x770, session_data)));
  CHECK(!tx(control(0, 3)));
  rx(packet(0x778, session_ack));
  CHECK(!tx(control(0, 3)));
  release(0);
  CHECK(!tx(control(1, 3)));
  release(1);
  CHECK(tx(control(0, 3)));
  response(0, 3);
  CHECK(!tx(control(1, 3)));
  timer.CNT = 10000;
  release(0);
  CHECK(!tx(control(1, 3)));
  release(1);
  timer.CNT = 200000;
  clear_input();
  CHECK(tx(control(1, 3)));

  for (int kind = 0; kind < 7; kind++) {
    reset();
    CHECK(tx(packet(0x770, session_data)));
    CANPacket_t p = packet(0x778, session_ack);
    if (kind == 0) p.bus = 2;
    if (kind == 1) p.returned = 1;
    if (kind == 2) p.rejected = 1;
    if (kind == 3) p.extended = 1;
    if (kind == 4) p.fd = 1;
    if (kind == 5) p.data_len_code = 7;
    if (kind == 6) p.addr = 0x770;
    rx(p);
    release(0);
    release(1);
    CHECK(!tx(control(0, 3)));
  }
  for (int kind = 0; kind < 4; kind++) {
    reset();
    CHECK(tx(packet(0x770, session_data)));
    rx(packet(0x778, session_ack));
    CHECK(tx(control(0, 0)));
    CANPacket_t p = packet(0x778, (uint8_t[8]){4, 0x6f, 0xbc, 0x15, 0, 0, 0, 0});
    if (kind == 0) p.data[3] = 0x16;
    if (kind == 1) p.data[4] = 3;
    if (kind == 2) p.data[0] = 5;
    if (kind == 3) { p.data[0] = 3; p.data[1] = 0x7f; p.data[2] = 0x2f; p.data[3] = 0x78; }
    rx(p);
    response(0, 0);
    release(1);
    CHECK(!tx(control(0, 3)));
  }
}

static void test_rate_limits_timeout_and_wrap(void) {
  reset();
  CHECK(tx(packet(0x770, session_data)));
  timer.CNT = 999999;
  clear_input();
  CHECK(!tx(packet(0x770, session_data)));
  timer.CNT = 1000000;
  CHECK(tx(packet(0x770, session_data)));
  reset();
  controls_allowed = false;
  CHECK(tx(control(0, 0)));
  CHECK(tx(control(1, 0)));
  timer.CNT = 9999;
  CHECK(!tx(control(0, 0)));
  timer.CNT = 10000;
  CHECK(tx(control(0, 0)));
  ready();
  CHECK(tx(control(0, 3)));
  response(0, 3);
  timer.CNT = 10000;
  release(0);
  release(1);
  timer.CNT = 199999;
  clear_input();
  CHECK(!tx(control(1, 3)));
  timer.CNT = 200000;
  CHECK(tx(control(1, 3)));
  reset();
  CHECK(tx(packet(0x770, session_data)));
  for (uint32_t t = 100000; t <= 500000; t += 100000) {
    timer.CNT = t;
    clear_input();
  }
  timer.CNT = 500001;
  rx(packet(0x778, session_ack));
  release(0);
  release(1);
  CHECK(!tx(control(0, 3)));
  reset();
  timer.CNT = UINT32_MAX - 5000U;
  CHECK(tx(control(0, 0)));
  timer.CNT = 4998;
  CHECK(!tx(control(0, 0)));
  timer.CNT = 4999;
  CHECK(tx(control(0, 0)));
}

static void test_existing_adas_decisions_unchanged(void) {
  const int addresses[] = {0x340, 0x4f1, 0x485, 593, 1056, 1057, 1290, 905, 909, 1155, 1186, 0x7d0, 0x7b1, 0x515, 0x526, 0x527};
  for (unsigned int i = 0; i < sizeof(addresses) / sizeof(addresses[0]); i++) {
    for (int bus = 0; bus <= 2; bus++) {
      for (int enabled = 0; enabled <= 1; enabled++) {
        CANPacket_t p = packet(addresses[i], (uint8_t[8]){0});
        p.bus = bus;
        if (p.addr == 0x4f1) { p.data_len_code = 4; p.data[0] = 1; }
        if (p.addr == 0x7d0) { p.data[0] = 2; p.data[1] = 0x3e; p.data[2] = 0x80; }
        reset_mode(SAFETY_HYUNDAI, 8U);
        controls_allowed = enabled;
        bool baseline = tx(p);
        reset_mode(SAFETY_HYUNDAI, 1024U | 8U);
        controls_allowed = enabled;
        CHECK(tx(p) == baseline);
      }
    }
  }
  ready();
  relay_malfunction = true;
  CHECK(!tx(control(0, 3)));  // Global safety dispatcher still applies.
  CHECK(!tx(control(0, 0)));
}

static void replay(void) {
  // time_us controls R|T address_hex bus flags data_hex expected_allowed
  // flags: fd=1, extended=2, returned=4, rejected=8. R uses the actual RX hook.
  CHECK(set_safety_hooks(SAFETY_HYUNDAI, 1024U | 8U) == 0);
  unsigned int timestamp, controls, address, bus, flags, expected;
  char direction, hex[17];
  int fields;
  unsigned int line = 0;
  while ((fields = scanf("%u %u %c %x %u %u %16s %u", &timestamp, &controls, &direction,
                         &address, &bus, &flags, hex, &expected)) != EOF) {
    line++;
    CHECK(fields == 8);
    CHECK((controls <= 1U) && (bus <= 7U) && (flags <= 15U) && (expected <= 1U));
    CHECK(strlen(hex) == 16U);
    CHECK((direction == 'R') || (direction == 'T'));
    uint8_t data[8];
    for (int i = 0; i < 8; i++) {
      unsigned int byte;
      CHECK(sscanf(&hex[2 * i], "%2x", &byte) == 1);
      data[i] = byte;
    }
    timer.CNT = timestamp;
    controls_allowed = controls;
    CANPacket_t p = packet(address, data);
    p.bus = bus;
    p.fd = (flags & 1U) != 0U;
    p.extended = (flags & 2U) != 0U;
    p.returned = (flags & 4U) != 0U;
    p.rejected = (flags & 8U) != 0U;
    can_set_checksum(&p);
    bool actual = direction == 'T' ? safety_tx_hook(&p) : safety_rx_hook(&p);
    if (actual != (expected != 0U)) {
      fprintf(stderr, "Trace line %u time=%u %c addr=%x data=%s expected=%u actual=%u\n",
              line, timestamp, direction, address, hex, expected, actual);
    }
    CHECK(actual == (expected != 0U));
  }
  CHECK(line > 0U);
  printf("HYUNDAI_ATC_REPLAY_EVENTS=%u\n", line);
}

int main(int argc, char *argv[]) {
  if ((argc == 2) && (strcmp(argv[1], "--replay") == 0)) {
    replay();
    return 0;
  }
  CHECK(argc == 1);
  test_opt_in_and_modes();
  test_exact_wire_shape();
  test_input_gates_and_invalidation();
  test_ack_sequence_and_echo_rejection();
  test_rate_limits_timeout_and_wrap();
  test_existing_adas_decisions_unchanged();
  printf("HYUNDAI_ATC_ACTUAL_HOOK_CHECKS=%d\n", checks);
  return 0;
}
