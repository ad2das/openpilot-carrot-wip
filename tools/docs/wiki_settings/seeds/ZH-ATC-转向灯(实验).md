<!-- CARROT:GENERATED schema="1" locale="zh" group="STEER_LANECHANGE" param="AutoTurnSignal" -->
<!-- CARROT:AUTHORING guide="https://github.com/ajouatom/openpilot/blob/carrot-wip/tools/docs/wiki_settings/AUTHORING_GUIDE.md" editable="CARROT:MANUAL only" -->
# ATC 转向灯（实验）

**菜单:** 驾驶控制 \> 转向 \> 变道 (自动转向)

---

<!-- CARROT:SETTING:BEGIN AutoTurnSignal -->
<a id="AutoTurnSignal"></a>

## ATC 转向灯（实验）

`AutoTurnSignal`

在使用传统 CAN 的帕里斯帝上，根据 ATC 转弯或岔路请求发送转向灯诊断脉冲。驾驶员操作、双闪、控制退出或数据过期会取消请求；诊断故障后锁定至重启。默认关闭，修改后须重新启动车辆或设备。实际闪烁、控制权归还及进程崩溃后的释放尚未验证。

<!-- CARROT:AUTO:BEGIN AutoTurnSignal -->
<!-- CARROT:AUTO:SOURCE facts="776f6ab63bb8423f19f514f5a96231975465a2c1180d0ddec45e55c47824482f" semantic="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" review="current" -->
<!-- CARROT:SECTION:BEGIN OVERVIEW -->
- **类型:** 开关
- **默认值:** `0`
- **范围:** `0` ~ `1`
- **菜单:** 驾驶控制 \> 转向 \> 变道 (自动转向)
<!-- CARROT:SECTION:END OVERVIEW -->
<!-- CARROT:AUTO:END AutoTurnSignal -->

<!-- CARROT:MANUAL:BEGIN AutoTurnSignal -->
<!-- CARROT:REVIEW reviewed-for="a66f79658c5c471a5fd2b42c126a84e68b6a2cd71bc6c44914fb2035d9dddc5f" reviewed-at="2026-10-26T07:59:29Z" -->
<!-- CARROT:SECTION:BEGIN BEHAVIOR -->
### 行为

开启后，根据传统 CAN 帕里斯帝的 ATC 转弯或岔路请求发送诊断激活及归还控制权脉冲。仅在驾驶与横向控制启用时开始；驾驶员操作、双闪、控制退出或数据过期会取消请求。关闭后不发送此类诊断指令。
<!-- CARROT:SECTION:END BEHAVIOR -->

<!-- CARROT:SECTION:BEGIN USAGE -->
### 使用

修改后须重新启动车辆或设备。诊断响应错误或超时后锁定至重启，已取消的同一转弯请求不会恢复。
<!-- CARROT:SECTION:END USAGE -->

<!-- CARROT:SECTION:BEGIN REFERENCE -->
### 参考

实际闪烁、驾驶员优先权、控制权归还及进程崩溃后的释放尚未通过实车验证。诊断响应不能证明转向灯实际工作。
<!-- CARROT:SECTION:END REFERENCE -->

<!-- CARROT:MANUAL:END AutoTurnSignal -->

<!-- CARROT:SETTING:END AutoTurnSignal -->
