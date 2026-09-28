# DisableDM 포크 자동 갱신

대상은 `ad2das/openpilot-carrot-wip`의 `carrot-wip`입니다. 원본은
`ajouatom/openpilot`의 `carrot-wip`이며, PR #524의 DisableDM 패치를 유지합니다.
이 자동화는 포크에만 설치하며 원본 PR 브랜치를 바꾸지 않습니다.

GitHub Actions가 매시 17분과 47분에 원본의 새 커밋을 확인합니다. 새 커밋을
기존 패치 브랜치에 merge하고, DisableDM·영상·검색·프리셋 회귀 검사와 웹 빌드가
성공한 경우에만 포크를 push합니다. 이미 포함된 원본에는 새 커밋을 만들지 않습니다.
패치를 매번 중복 삽입하지 않고 Git 병합으로 유지합니다.

충돌, 패치 기능 검사 실패, 빌드 산출물 불일치, push 권한 오류가 있으면 기존
원격 브랜치를 유지합니다. 검사 중 사용자가 브랜치를 갱신해도 강제로 덮어쓰지
않습니다. 원본 구조가 바뀌어 충돌하면 수동으로 한 번 수정한 뒤 다시 실행해야 합니다.
원본이 GitHub 워크플로 파일을 바꿔 GitHub 토큰의 쓰기 제한에 걸리는 경우에도
수동 검토가 필요할 수 있습니다. 테스트 통과는 차량 주행 검증을 뜻하지 않습니다.

- 자동 갱신 브랜치: https://github.com/ad2das/openpilot-carrot-wip/tree/carrot-wip
- 실행·실패 기록: https://github.com/ad2das/openpilot-carrot-wip/actions/workflows/disabledm-sync.yml
- 즉시 확인: 위 Actions 화면에서 **Run workflow**를 실행합니다. 수동 실행은 새 원본이 없어도 검사를 수행합니다.
- 중지: 해당 워크플로를 **Disable workflow**로 비활성화합니다.

GitHub 예약 실행은 부하에 따라 늦어질 수 있으며, 공개 저장소가 60일 동안
비활성 상태이면 GitHub가 예약을 중지할 수 있습니다. 이 경우 Actions에서 다시
활성화합니다. [GitHub 예약 실행 문서](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

이 PC의 폴더와 차량은 이 작업으로 자동 pull하거나 재부팅하지 않습니다.
차량에서 이 포크를 사용하려면 설치/업데이트 대상 저장소와 브랜치를 위 대상으로
선택해야 합니다. 저장된 DisableDM 값과 재부팅 적용 방식은 유지됩니다.

## 구현 및 검사

`sync.py prepare`는 깨끗한 임시 CI checkout에서 후보 merge 커밋을 만듭니다.
검사 성공 뒤 별도 `publish` 단계가 동일 후보와 원격 기준 커밋을 확인하고
일반 push를 수행합니다. force push와 자동 충돌 우선순위는 사용하지 않습니다.
쓰기 인증 정보는 마지막 publish 단계에만 설정합니다.

`test_sync.py`는 실제 임시 Git 저장소로 정상 갱신, 반복 실행, 충돌 복구,
검사 실패 시 미게시, 동시 갱신 거부, 미저장 작업 보존을 검사합니다.
