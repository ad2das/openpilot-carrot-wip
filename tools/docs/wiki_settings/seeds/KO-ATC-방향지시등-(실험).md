<!-- CARROT:GENERATED schema="1" locale="ko" group="STEER_LANECHANGE" param="AutoTurnSignal" -->
<!-- CARROT:AUTHORING guide="https://github.com/ajouatom/openpilot/blob/carrot-wip/tools/docs/wiki_settings/AUTHORING_GUIDE.md" editable="CARROT:MANUAL only" -->
# ATC 방향지시등 (실험)

**메뉴:** 주행 제어 \> 차량 조향 \> 차로 변경 (자동 턴)

---

<!-- CARROT:SETTING:BEGIN AutoTurnSignal -->
<a id="AutoTurnSignal"></a>

## ATC 방향지시등 (실험)

`AutoTurnSignal`

팰리세이드 일반 CAN에서 ATC 회전·분기 요청에 따라 방향지시등 진단 펄스를 송신합니다. 운전자 조작·비상등·제어 해제·오래된 입력은 취소하며 진단 오류 후에는 재시작까지 잠깁니다. 기본 꺼짐. 변경 후 시동 또는 장치를 재시작해야 합니다. 실차 점멸·제어권 반환·프로세스 종료 시 해제는 미검증입니다.

<!-- CARROT:AUTO:BEGIN AutoTurnSignal -->
<!-- CARROT:AUTO:SOURCE facts="991c08d99654c252fb03380641de7ed6e393db54d208b0f951e6127d52681111" semantic="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" review="current" -->
<!-- CARROT:SECTION:BEGIN OVERVIEW -->
- **종류:** 토글
- **기본값:** `0`
- **범위:** `0` ~ `1`
- **메뉴:** 주행 제어 \> 차량 조향 \> 차로 변경 (자동 턴)
<!-- CARROT:SECTION:END OVERVIEW -->
<!-- CARROT:AUTO:END AutoTurnSignal -->

<!-- CARROT:MANUAL:BEGIN AutoTurnSignal -->
<!-- CARROT:REVIEW reviewed-for="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" reviewed-at="2026-10-26T07:59:29Z" -->
<!-- CARROT:SECTION:BEGIN BEHAVIOR -->
### 동작

켜면 일반 CAN 팰리세이드의 ATC 회전·분기 요청에 따라 진단 활성·반환 펄스를 송신합니다. 주행·조향 제어가 활성화된 경우만 시작하며, 운전자 조작·비상등·제어 해제·오래된 입력은 취소합니다. 끄면 이 진단 송신을 사용하지 않습니다.
<!-- CARROT:SECTION:END BEHAVIOR -->

<!-- CARROT:SECTION:BEGIN USAGE -->
### 사용

설정 변경은 시동 또는 장치를 재시작한 뒤 적용됩니다. 진단 응답 오류나 시간 초과가 발생하면 재시작까지 잠기며, 취소된 같은 회전 요청은 다시 시작하지 않습니다.
<!-- CARROT:SECTION:END USAGE -->

<!-- CARROT:SECTION:BEGIN REFERENCE -->
### 참고

실제 점멸, 운전자 우선순위, 제어권 반환과 프로세스 비정상 종료 후 해제는 실차 미검증입니다. 진단 응답을 램프 작동 성공으로 판단하지 마세요.
<!-- CARROT:SECTION:END REFERENCE -->

<!-- CARROT:MANUAL:END AutoTurnSignal -->

<!-- CARROT:SETTING:END AutoTurnSignal -->
