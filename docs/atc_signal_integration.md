# Experimental Palisade ATC turn signals

The navigation request pipeline, runtime CAN sender, cancellation/cleanup,
Panda command restrictions and `AutoTurnSignal` setting are implemented.
The setting defaults off and applies at the next onroad start, only to the
classic Hyundai Palisade fingerprint with Hyundai nonlegacy safety.
**Software verification does not establish physical turn-signal operation.**
The vehicle was unavailable for the final actuator checks. Command ACKs and
`CarState.atcSignalStatus` describe transactions, never proof that a lamp lit.

## Implemented path

```text
actual navigation item receipt
  -> CarrotServ maneuver episode + source-change cancellation
  -> carrotMan.atcSignal
  -> DesireHelper resolved navigation provenance + modeld arbiter
  -> modelV2.meta.atcSignal
  -> controlsd current-input arbiter
  -> carControl.atcSignal (including control publication timestamp)
  -> Palisade controller: opt-in + fresh control + live-input arbiter
  -> diagnostic transaction/pulse state machine
  -> sendcan -> exact Panda safety hook -> 0x770 bus 0
  <- actual received 0x778 bus 0 -> CarInterface reply snapshot
```

The producer requires a fresh `none` interval of one second before assigning a
new episode. Prepare, turn and fork phases share an episode. Cancellation,
source loss and a direction reversal cannot be cleared by another active
packet. Legacy sender identity, TCP navigation session identity and selection
of `navInstruction` are tracked separately from the old external-navigation
boolean. A change cancels the current episode even if navigation remains alive.

Freshness uses the legacy TBT packet receipt, TCP current-guidance item receipt,
or selected `navInstruction` event timestamp. Repeated publication and unrelated
GPS/speed-camera packets do not renew an old maneuver. The navigation age limit
is two seconds; source and local receive times of intermediate services are
also checked. These are conservative software bounds, not measured ECU limits.

Both consumers reject stale, malformed and out-of-order identities. Startup
with an already-active maneuver is blocked. A genuine preparation message can
establish an inactive baseline; a synthetic `helper_not_ready` message cannot.
Current driver switch, hazard, lateral disengagement, invalid CAN, trailer,
opposite torque and stale input stop the candidate before old-ID filtering.
Opposite torque is evaluated using the original navigation direction even
when the helper's output has been narrowed to a zero-direction waiting state.

The helper publishes navigation provenance only after its existing maneuver
logic resolves the request. Bluetooth and legacy `LANECHANGE` commands cannot
inherit that provenance. A turn can carry a request with lane-change state
`off`; a fork can carry `fork_pending` while model desire is still `none`.
The existing vehicle blinker output fields retain their previous behavior.
With the new feature off, the helper's driver-blinker behavior is unchanged.
When enabled, the helper suppresses lamp feedback only during an automatic
transaction and until its public lamp hold ends. A real switch press restores
the original manual lamp-based path, including one-touch holds. The interface
retains the press timestamp across 100 Hz CAN / 20 Hz model boundaries, even
if press and release arrive in the same receive batch. A recent press cancels
the ATC episode. An already active OEM lamp prevents a new session. Unknown
input or a hazard signal blocks automatic manoeuvre signalling.

## Runtime output and cleanup

The controller independently checks request publication age (150 ms), original
navigation age (2 s), raw switches (200 ms), enabled lateral control, valid
CAN, drive/low gear, no steering fault, no trailer and no opposite driver
torque. Startup active requests are quarantined; recovery cannot revive a
canceled episode. Fresh input is evaluated before any received ACK.

Each maneuver starts `10 03`, waits for a matching `50 03`, then returns both
sides to ECU control with `2F BC 15 00` and `2F BC 16 00`, each acknowledged
before the next command. Only then may one side receive `03`. The nominal
software pulse is 350 ms from the ON command, both return-control commands
and ACKs, then at least 350 ms OFF before the next ON. No flash behavior is
assumed from `03`. These intervals are software policy, not a measured lamp
waveform or hard realtime guarantee.

ACK timeout is 250 ms, with at most three release attempts per side. A
negative, malformed, unexpected or timed-out response latches a fault until
process/ignition restart, clears queued ON and attempts both releases. There
is no ON retry after a fault. A maneuver is bounded to 45 seconds; a possibly
active command also has a one-second watchdog. A clean end discards the
session, so a new maneuver requires the full handshake again.

Only actual DLC-8 receive frames from bus 0/address 0x778 are consumed. Returned
or rejected Panda frames are not ACKs. Each batch is drained once, and an ACK
already received cannot acknowledge a command generated during that update.
`atcSignalStatus` records phase, acknowledged direction, fault, command/reply
times and command count; none of these confirms physical lamp state.

When carControl dies or initialization is lost, `card` runs an ATC-only cleanup
path without applying stale steering/acceleration commands. Invalid control
events also block acquisition. A graceful process exit attempts both
return-control commands once. Process crashes, power loss, stopped CAN or a
stopped Panda cannot guarantee delivery or physical release.

Panda opt-in bit 1024 permits only exact classic bus-0, DLC-8, zero-padded
`10 03` and `2F BC 15/16 03/00`. Session/ON additionally require controls
allowed, clear fresh CGW1 inputs, a matching session ACK and both release
ACKs. Each ON consumes both release permissions. Minimum intervals are one
second per session, 200 ms per ON and 10 ms per side's release. Driver input,
staleness or disengagement invalidate acquisition proof. Return-control
remains permitted during disengagement for cleanup. Other ADAS TX policies
are unchanged. This safety hook restricts host output; it cannot send cleanup
after a dead host or prove ICU support.

## Receive data and diagnostic evidence

On the Palisade fingerprint only, `CarState.turnSignalInput` records CGW1 bus-0
0x541 switch candidates at bits 42/41, hazard bits 33–34, and lamp bits 19–20
and 62–63. Sample age, bus, size and reserved values are checked. The adapter
does not read `CANParser.can_valid`: that property consumes the existing
invalid-frame debounce. The interface retains its sole read; candidate
consumers separately check the published `CarState.canValid`.

`turnSignalInput.valid` means fresh and decodable, **not proof that switch
signals remain independent while a diagnostic command operates the lamps**.
Public lamp booleans retain the existing 50-frame hold and are not interpreted
as driver acknowledgement or a diagnostic response.

Static ScanDoc Palisade LX2 data for five ICU profiles binds request/response
0x770/0x778, session `10 03`, left `2F BC 15 03` and right `2F BC 16 03`.
The respective `00` controls return control; they do not prove physical OFF.
The pinned source hashes are in `hyundai/atc_diagnostics.py`. This is a
third-party diagnostic-tool source, not vehicle acceptance evidence.

The transaction planner requires explicit bus/timing parameters. It
matches real-receive-source diagnostic replies, rejects Panda return echoes,
and separates acknowledgement from physical lamp state. It requests both
releases before a direction change, bounds release retries, clears stale
queued directions and latches faults. No automatic ON retry, inferred tester
present cadence, capability-query `02`, or guessed session-close command was
added. A new instance starts with unknown ownership.

## Validation

Focused software tests (normal project environment):

```sh
python -m pytest -q \
  openpilot/selfdrive/controls/lib/tests/test_atc_signal.py \
  openpilot/selfdrive/controls/lib/tests/test_atc_signal_integration.py \
  opendbc_repo/opendbc/car/hyundai/tests/test_atc_diagnostics.py \
  opendbc_repo/opendbc/car/hyundai/tests/test_atc_blinker.py \
  opendbc_repo/opendbc/car/hyundai/tests/test_classic_atc_interface.py
python opendbc_repo/opendbc/safety/tests/test_hyundai_atc_c.py
```

The final focused run passed 299 tests and 31 subtests, including the existing
manual lever/turn tests. The integration tests exercise the actual helper, geometry and classifier,
real cereal serialization, producer/model/control adapters, actual controller
CAN output and diagnostic exchanges. Only platform Params and Bluetooth I/O
are replaced on the Windows host.
They cover both directions, pending/active forks, driver/hazard cancellation,
first-frame and between-publication torque, source replacement, restart,
staleness, malformed input and response/release failures. The enabled and
disabled actual controllers produce identical ADAS frames. Production runtime
traces (normal pulses, lost controls and negative ECU replies) are also fed
through the compiled Panda dispatcher. GCC/UBSan executes the real C hooks in
both default and ALLOW_DEBUG configurations, including 1,054 checks per build.
Tests do not start the full vehicle process graph or exercise physical CAN.

Scope tests cover all 107 Hyundai platform identifiers. With forced host and
safety bits, the runtime gate accepts only Palisade. Actual parameter generation
succeeds for 106 platforms in both setting states; all non-ATC parameter values
are identical. The remaining KIA_K5_DL3_24_HEV platform has an existing torque
data key mismatch at baseline `05c1678033c1c778f0386fba00c1e105bb26c427`, producing
the same KeyError with the setting off/on. That unrelated steering data was not
changed; its direct runtime fingerprint gate was still tested.

Read-only replay of three existing Palisade segments decoded 1,934 CGW1
batches and compared 9,670 fields with the real DBC parser, with zero
differences and no extra consumption of its CAN-error debounce. The new
fields defaulted inactive in 24,879 historical navigation/model/control
messages. The final replay uses the latest full cereal schema and decodes all
events; the earlier checkpoint's 119 tag-152 events now have their schema.

A deterministic 1,800-frame comparison against baseline
`05c1678033c1c778f0386fba00c1e105bb26c427` compared 25 existing helper state
values per frame: 45,000 exact comparisons, zero differences. This protects
existing behavior when disabled; it is not evidence of physical lamp control.
Separate opt-in regressions verify manual one-touch parity, automatic lamp
feedback suppression and retained press timestamps across model frames.

Earlier reproducibility artifacts and SHA manifests are indexed under
`.analysis/archive/2026-10-26-atc-code/` in the workstation workspace.
The earlier complete 898-file log scan and pinned vendor extraction remain in
`.analysis/archive/2026-10-26-atc-log-protocol/`. No private route files are
included in Git.

## Remaining physical dependency

The existing 898 logs contain no accepted 0x770/0x778 body-control transaction.
They cannot establish the connected ICU's runtime actuator availability,
vehicle-bus routing, whether `03` is a steady lamp or a flashing program,
moving-speed acceptance, independent driver override, release behavior,
session timeout, or process/power-loss cleanup. The experimental sender uses
the statically identified commands with strict response gates and defaults
off. An opt-in setting and passing simulations do not supply this missing
physical evidence. Software implementation, repository delivery and vehicle
validation must be reported separately.
