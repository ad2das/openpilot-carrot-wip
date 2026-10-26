<!-- CARROT:GENERATED schema="1" locale="en" group="STEER_LANECHANGE" param="AutoTurnSignal" -->
<!-- CARROT:AUTHORING guide="https://github.com/ajouatom/openpilot/blob/carrot-wip/tools/docs/wiki_settings/AUTHORING_GUIDE.md" editable="CARROT:MANUAL only" -->
# Experimental ATC Turn Signals

**Menu:** Driving \> Steering \> Lane Change (Auto Turn)

---

<!-- CARROT:SETTING:BEGIN AutoTurnSignal -->
<a id="AutoTurnSignal"></a>

## Experimental ATC Turn Signals

`AutoTurnSignal`

Send diagnostic turn-signal pulses for ATC turns and forks on classic-CAN Palisade. Driver input, hazards, disengagement or stale input cancel the request; a diagnostic fault locks it until restart. Off by default. Restart the vehicle or device after changes. Physical flashing, return of control and release after a process crash remain unverified.

<!-- CARROT:AUTO:BEGIN AutoTurnSignal -->
<!-- CARROT:AUTO:SOURCE facts="cc528c935a471c7fa7a7756fce57a8539dcffaa814bf8feefbb1ad7560d55dbf" semantic="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" review="current" -->
<!-- CARROT:SECTION:BEGIN OVERVIEW -->
- **Type:** Toggle
- **Default:** `0`
- **Range:** `0` ~ `1`
- **Menu:** Driving \> Steering \> Lane Change (Auto Turn)
<!-- CARROT:SECTION:END OVERVIEW -->
<!-- CARROT:AUTO:END AutoTurnSignal -->

<!-- CARROT:MANUAL:BEGIN AutoTurnSignal -->
<!-- CARROT:REVIEW reviewed-for="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" reviewed-at="2026-10-26T07:59:29Z" -->
<!-- CARROT:SECTION:BEGIN BEHAVIOR -->
### Behavior

When enabled, send diagnostic activation/return pulses for ATC turns and forks on classic-CAN Palisade. Starting requires active driving and lateral control; driver input, hazards, disengagement or stale input cancel the request. Turning it off disables this diagnostic transmission.
<!-- CARROT:SECTION:END BEHAVIOR -->

<!-- CARROT:SECTION:BEGIN USAGE -->
### Usage

Changes apply after restarting the vehicle or device. A diagnostic response error or timeout locks the feature until restart, and a canceled maneuver cannot resume.
<!-- CARROT:SECTION:END USAGE -->

<!-- CARROT:SECTION:BEGIN REFERENCE -->
### Reference

Physical flashing, driver priority, return of control and release after a process crash remain unverified on a vehicle. A diagnostic response does not prove the lamp operated.
<!-- CARROT:SECTION:END REFERENCE -->

<!-- CARROT:MANUAL:END AutoTurnSignal -->

<!-- CARROT:SETTING:END AutoTurnSignal -->
