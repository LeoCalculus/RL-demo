# Plan mode

Q: Given I wanna make robot stand with disturbance, here is a design for reward I haven’t implement the formula yet, can u check and give me suggestions on these design, like limitations? any potential reward hacks? reward flow chat @reward_design.drawio

A:

Yes, we can define the reward criteria first, then use them to decide which measurements the environment needs. We still need a few task choices alongside the rewards: allowed motion, reference frames, contact rules, and what counts as recovery. Final weights will need trials in the environment.

Your diagram covers height, body attitude, unwanted support contacts, posture, and motor effort. The following are possible loopholes, not behavior already observed from a trained policy.

| Design item | Limitation or possible reward hack | Suggestion |
|----|----|----|
| Maintain height | The robot could keep the target height while rolling away, spinning, or bouncing through the target. | Define height at one fixed body/site relative to the ground. Decide whether displacement and heading matter; consider residual linear/angular velocity and a sustained recovery check. |
| Main body plane parallel to ground | A plane can be parallel while upside down. A level body also does not guarantee an acceptable leg/arm pose. | Use an orientation measure that distinguishes upright from inverted. Treat selected joint posture separately. |
| Height inside tolerance | A simple pass/fail reward gives no preference among states inside the band and little guidance outside it. It can tolerate repeated crossings. | Compare a smooth error reward with a deadband cost that increases outside tolerance. Check vertical velocity and time spent recovered. |
| Use wheels only | This forbids other support surfaces, but does not specify whether all four wheels must remain down. Hopping or balancing on fewer wheels might qualify. | Decide whether temporary wheel lift is allowed. Detect non-wheel ground contact across all relevant collision geoms. |
| Keep appearance unchanged | A strong penalty on every joint change can prevent the leg adjustment needed to recover. Continuous wheel angles also change during valid rolling. | Identify which hinges should stay near reference and which may adapt. Exclude wheel spin angles and avoid counting passive linkage motion as independent behavior. |
| Maintain height mainly through leg linkage | Rewarding leg movement itself could encourage needless cycling; height alone does not establish which mechanism caused recovery. | Describe this as allowed joint motion and restricted arm/folding motion. Add another reward only if a measurable behavior is missing. |
| Torque and speed limits | A soft penalty does not enforce a hard limit. Excessive effort penalties can favor collapse or weak recovery. | Keep physical limits in actuator/control settings; use rewards for efficiency within the permitted behavior. Evaluate recovery and effort together. |
| Sustained high torque | A single-step cost does not distinguish a brief recovery effort from prolonged loading. Net signed torque can cancel across joints. | Decide whether you mean instantaneous effort or a time-window load measure, using per-motor ratings. Any history must reset per episode; decide how the controller observes relevant history. |
| Penalty and restart | If continuing produces enough negative return, early failure may become preferable. A contact threshold or delay could also allow brief support on another limb. | Compare returns for successful recovery, prolonged poor behavior, and early failure. Specify contact thresholds and timing explicitly. Keep failure separate from episode timeout. |

For wheels, normal rolling is not the same as sliding. If you later add a slip cost, use wheel-ground relative motion at contact; penalizing wheel-center horizontal velocity would also penalize valid rolling.

I suggest starting with a small set of terms: height, uprightness, settling motion, selected posture, and modest effort/action-smoothness costs. These are proposals, not approved formulas or weights. Keep non-wheel support as a failure condition if that is your intended rule. Evaluate height and attitude throughout recovery rather than hiding them whenever another condition fails.

Two useful alternatives:

* **Smooth weighted terms (suggested starting point):** provide feedback during imperfect recovery. Their scales must be checked so posture or efficiency does not outweigh recovery.
* **Success bonus only when every tolerance is met:** expresses the final goal clearly, but gives little guidance to a policy that rarely succeeds. A sustained-success metric can be used for evaluation even when training uses smooth terms. Avoid repeatedly awarding a one-time recovery bonus on each threshold crossing.

There is one framework detail to account for later: mjlab 1.6.0 weights each reward term and, by default, multiplies it by the environment step duration. Specify whether tests expect the raw term or its final weighted contribution, and distinguish per-time costs from one-time failure penalties. [Reward manager source](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/managers/reward_manager.py).

Also, its built-in upright reward measures squared horizontal projected gravity on flat ground. My inference from that formula is that upright and fully inverted poses receive the same value from this term alone; a separate fall condition may prevent the inverted state from persisting. Do not assume the term alone checks the intended orientation. [Upright reward source](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/mdp/rewards.py).

Before writing formulas, settle these points:


1. Does recovery permit rolling to a new position, yaw rotation, or temporary wheel lift?
2. Which body/site defines height, which ground reference is used, and is the target fixed or taken from the reset pose?
3. Which joints may adapt, and which arm/folding joints should remain near reference?
4. What height, tilt, and motion tolerances must hold, for how long, and by what recovery deadline?
5. What motor ratings, load duration, and exact failure/contact rules should be used?

Then write one Implementation mode note per calculation with its destination, formula, inputs/frames/units, joint selection, parameters, boundary behavior, reset behavior if stateful, and numerical tests with expected outputs. Include examples for upright versus inverted, height at and outside tolerance, wheel-only versus forbidden contact, and brief versus sustained effort. These examples will help expose loopholes before training.


---

Q: Actually I think maintain height should change, since  I have inital angle as joint variables, and I actually wanna them with external force can still be settle which means the robot can recover to original joint variables, and height is kinda reference we can use, like persuring the reward, dispalcement for the robot does not matter, what it matters to keep height and keep standing not falling.

Main body plane parallel to ground: make sure it is not inclined, so all gravity component captured by IMU at z axis should be -9.81 point down, close to this gets more reward.

Temporary lift others all forbidden only wheels allowed, other touch => considered as failure

hinge should finally goes back to that pos, it can temporary used for assit with balance but in long term in settle expect to go to original pose

Must use critic to ensure at settle time the wheels on ground or penalty

Maybe a soft one like once the torque enter 0.9 of limit them reduce little reward may be preferred.

high torque I mean the robot stuck at its limit for like 5s give giant penalty

A:

This makes the goal clearer: **after a disturbance, return to the reference joint pose, remain upright, recover the target height, and settle with wheel support.** Horizontal displacement does not need a penalty. Returning joint angles alone does not guarantee height or balance because the whole robot can translate and rotate.

I suggest the following structure. These are planning suggestions, not final formulas or weights.

| Part | Suggested treatment |
|----|----|
| Joint pose | Compare the selected hinges with a fixed reference pose. Allow temporary deviation during recovery. Exclude continuous wheel spin angles; specify the leg, arm, and folding joints included. |
| Height | Keep a separate height error as a supporting objective. Choose its body/site, ground reference, and target rather than assuming joint angles determine it. |
| Uprightness | Measure signed gravity direction in the IMU frame so inverted is worse than upright. Keep this active during recovery. |
| Settling | Check small joint and body velocities over a sustained interval. This permits displacement during recovery without rewarding endless rolling or oscillation. Decide separately whether yaw angle matters. |
| Support | Any non-wheel ground contact is failure, as you specified. Check each wheel's contact independently for the settled-support requirement. Clarify whether wheel lift during recovery is allowed. |
| Motor load | Use a small cost near the effort limit and a separate per-motor duration measure for the five-second condition. Define which motors and which limits apply. |

**About the IMU:** your `(0, 0, -9.81)` target describes gravity expressed in an upright frame, assuming gravity magnitude `9.81 m/s²`. A raw accelerometer also responds to motion during a push, so it is not a pure tilt measurement. I suggest orientation-derived gravity for the training reward and an IMU attitude estimate for the deployed controller. MuJoCo describes accelerometer output in the sensor's local frame. [MuJoCo sensor reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer).

In mjlab 1.6.0, the root projected-gravity quantity uses a unit vector: upright is `(0, 0, -1)`, not `(0, 0, -9.81)`. Use the signed z error or full-vector error, rather than squared horizontal components alone. The root quantity only matches the IMU frame when their axes align; your current model defines them aligned. [Gravity initialization](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/entity/entity.py), [projection calculation](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/entity/data.py). Your current model adds a gyro and velocimeter, but no accelerometer.

**About the critic:** give it clean per-wheel contacts and other useful state, but enforce the contact requirement through the reward/failure calculation. The critic learns expected return; it does not itself guarantee that wheels stay down. Avoid defining “settled” to require wheel contact and then penalizing missing contact only when settled: that would switch off the penalty exactly when a wheel is missing. Instead, choose a recovery deadline/window or an independently defined settling condition, and check contacts there.

For posture, a weak continuous reference-pose cost is simpler and gives feedback throughout recovery, but can resist useful motion. A stronger cost after a specified recovery window allows more temporary movement, but needs timing rules and can encourage delaying recovery. I suggest starting with the continuous version and evaluating this tradeoff.

For torque, compare each motor's absolute applied torque with its own limit. Treat the suggested `90%` soft-cost onset separately from the threshold that means “stuck at the limit.” Define whether five seconds means continuous saturation, whether the large penalty fires once or continues, and when it rearms. A continuous timer can be avoided by dipping below threshold just before five seconds; accumulated exposure with gradual decay is an alternative, but measures a different requirement. Timers must reset per environment, and their state/history should be considered for observations.

Next, specify:


1. Reference joints/angles and height target; tolerances for pose, height, attitude, and velocities.
2. Whether wheel lift is allowed during recovery, whether all four wheels must support at settlement, and the settling duration/deadline. Are penalties after that deadline permanent until recovery, or assessed in another way?
3. Contact detection rules, soft-cost shape/size above `90%`, saturation threshold, exact five-second boundary, penalty size, and timer reset/rearm rules.

Then write the first Implementation mode note for a single calculation, such as uprightness or reference-pose error, with its formula and numerical expected test outputs. No Python changes are needed for this planning step.


---

Q: For Support im a bit disagree now, since when the force apply the wheel definitely will be off ground for this case will be immediately consider as failure, I only consider this may be after a after a few seconds? not all the time at least.

height might need to calculated from model and given angles first @p1_rl.xml

yeah all four wheel must used for support

if contact is not wheel -→ fail

I also add rolling in drawio, check what else specificaton is missing from drawio

A:

Agreed: **a wheel leaving the ground during recovery should not cause immediate failure.** Keep these rules separate:

* Non-wheel robot contact with the ground: failure, as you specified.
* Missing wheel contact during the recovery window: allowed.
* After that window: all four wheels must support the robot. Specify whether missing support then gives a penalty, failure, or a penalty followed by failure after a further delay.

For timing, a **fixed window after the external push ends** is simple to evaluate. A **timer starting when wheel contact is lost** also handles unexpected lift, but brief recontact can repeatedly reset it. I suggest the first option for scheduled training pushes. Neither should depend on the robot declaring itself settled, since it could avoid the support check by never settling. Specify how a second push affects an unfinished recovery window.

“Support” also needs a measurement: mere contact detection or a minimum upward contact force per wheel. Touching with almost no load can satisfy a contact-only rule. Four-wheel support does not require equal force on all four wheels.

**Height from the model:** yes, calculate a fixed reference using the reference angles, upright base orientation, wheel geometry, and ground plane. Do not recompute the target from the current moving joints each step; that would let the target follow an unwanted pose.

I made a preliminary analytical calculation from `robot_model/mujoco/p1_rl.xml` and `STAND_ANGLE_DEG`, applying the CAD-to-base conversion and passive linkage relations. With folding joints and unspecified joints at zero, the base-origin heights at which the tire cylinders touch a horizontal plane are:

| Wheels | Base height for geometric contact |
|----|----|
| Front pair | Approximately `0.485730 m` |
| Rear pair | Approximately `0.485416 m` |

Their difference is about `0.314 mm`. At exactly fixed angles and perfectly level attitude, vertical translation alone cannot put all four ideal rigid tire surfaces exactly on the plane. This is small, but the final pose/contact tolerances must allow a consistent solution. The existing spawn height `0.487424 m` leaves small geometric gaps. These estimates concern the **base origin**, not the IMU height, and have not been verified with loaded dynamics or other-body collision checks. MuJoCo is unavailable in the current Python; confirm using the compiled transformed model before fixing a reward target. [MuJoCo forward-kinematics reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-kinematics).

I checked the updated diagram. The rolling and recovery-speed branches need these additions, along with the earlier branches:

| Item | Specification still needed |
|----|----|
| Rolling | Does this mean body roll/pitch rotation or wheel-driven translation? “Acceleration about x/y” is ambiguous. Angular acceleration misses constant angular velocity; linear acceleration misses constant-speed rolling and also reacts to the external push. For body rocking, consider roll/pitch angular velocity alongside attitude error. For continued translation, consider horizontal velocity near settlement. |
| Recovery speed | Define when timing starts, the exponential decay rate, what counts as recovery, and how long success must persist. If this is a recovery bonus, issue it once per external disturbance; repeated threshold crossings must not earn repeated bonuses. Keep feedback for recovery that happens late. |
| Joint return | List the exact joints, reference angles, allowed errors, and joint-speed tolerances. “Appearance should not change” still needs to distinguish permitted temporary movement from the final pose requirement. |
| Height/uprightness | Specify the measured body/site, reference frame, error formula, tolerance, and weight. Make upright distinguishable from inverted. |
| Support/failure | Add the recovery-window duration, per-wheel support definition, required support duration, and the consequence after the deadline. State ground contact separately from self-contact and any excessive-tilt failure. |
| Motor load | Add per-motor torque/speed limits, the soft cost above `90%`, and the five-second saturation threshold, penalty, and timer reset/rearm rules. |
| Combined reward | Specify term signs, weights, and which terms apply during recovery versus after the deadline. The diagram's branches should not silently turn other useful terms off. |

I suggest settling the **support timing rule and meaning of rolling next**, then defining recovery success. Those choices determine the recovery-time calculation and which measurements its reward needs.


---

Q:

recovery time I suggest 3s at max, consider we have 30 rpm motor we even only need 2s to make full 360 deg rot

for joint return most hip motors shoud be original angles otherwise shape will have a lot of change

for height actually I think we can use something from critic like center of mass height that will be easier

for motors I have got motors.txt which u can use directly

anything not clear?

A:

Yes, a few design choices remain. **Three seconds is now your proposed maximum recovery time**, and `motors.txt` supplies the motor mapping and adopted simulation limits. You do not need to copy those values into another note.

At a constant `30 rpm`, one revolution takes two seconds, but recovery also requires acceleration, braking, balance, and contact recovery under load. The file lists `30 rpm` as the RH-17's no-load speed, not a common operating speed for all joints. Use three seconds as a task requirement to evaluate, rather than a guarantee from motor speed.

From `motors.txt`, section 3:

| Motor/group | Rated torque | Adopted simulation torque cap | Rated speed |
|----|----|----|----|
| RH-17 / folding | `35 N·m` | `54 N·m` | `25 rpm` |
| DM6248P / leg joints | `30 N·m` | `97 N·m` | `40 rpm` |
| M3508 + C620 / wheels | `3 N·m` | `4.5 N·m` stall reference | `469 rpm` |
| DM4340P / arms | `12 N·m` | `40 N·m` | `36 rpm` |

These distinguish continuous ratings from the adopted caps. Your five-second penalty is a task rule; the file does not establish that peak torque is physically permissible for five seconds. Its older viewer-controller settings also do not automatically apply to the current mjlab configuration; resolve motor mapping by joint name rather than copying actuator IDs.

**Center-of-mass height is available from simulator state; the critic can receive it.** Use that measured state directly for the reward, not a value predicted by the critic. There are two choices:

* **Whole-robot CoM:** includes the legs and arms. Useful for overall mass distribution, but moving limbs can change it while the chassis stays at the same height.
* **Base-body CoM or a fixed chassis site:** follows the main body's height more directly. I suggest this if maintaining the chassis height is the goal.

MuJoCo's `subtreecom` at the robot root gives whole-robot CoM, whereas mjlab's `root_com_pose_w` uses the root body's own CoM. They are different quantities. [MuJoCo subtree CoM](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-subtreecom), [mjlab root CoM implementation](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/entity/data.py). Either choice still needs a fixed target from the intended standing pose and ground reference. The earlier `0.4857 m` estimate is a base-origin height, not a CoM target.

The remaining questions are:


1. **Three-second rule:** does timing start when the push starts or ends? Must recovery be sustained for a short interval within those three seconds? At the deadline, is incomplete recovery a penalty or immediate failure? I suggest timing from push end for a finite-duration push.
2. **Joint selection:** does “hip motors” mean the four `*_hip_swing` joints, or all 12 leg motors including thigh and knee? Specify arm/folding behavior too. Reference angles already exist; the joint selection and allowed errors remain to be chosen.
3. **Height:** whole-robot CoM, base-body CoM, or a chassis site? Choose the target/reference-pose procedure and tolerance.
4. **Support and settling:** choose the minimum contact/load per wheel, required support duration, and pose/attitude/velocity tolerances. Non-wheel ground contact remains failure throughout recovery.
5. **Rolling:** body roll/pitch motion or horizontal wheel-driven movement? This is still ambiguous in the diagram.
6. **Torque timing:** which cap defines `90%` and “at the limit,” and is five seconds continuous or accumulated? If unresolved recovery terminates at three seconds, a saturation timer starting with that push cannot reach five seconds before termination. Decide whether that is intended, and specify penalty/reset behavior.

After these choices, the remaining work is writing the formulas, weights, and numerical test expectations in Implementation mode. We can define one term at a time; the full reward does not need to be specified in a single note.


---

Q: 1. if full im not sure about the force, if reasonable I will go with overall 5s


2. hip only for those 4, the arm should be 150 and -150 all the time lift up not touching the ground
3. whole body
4. 90% time of 5s on ground and 10% allow in air used for adjust pose
5. remove this
6. continuous of course, sometimes need large torque is allowed

if you think its ok, update the drawio so we can check

A:

This is a reasonable draft to review. Five seconds is a task target; we cannot judge whether a push is recoverable until its force, direction, application point, and duration are specified.

Updated [reward_design.drawio](reward_design.drawio) and added a [PNG preview](reward_design.preview.png). The diagram now shows:

* Overall recovery window: **5 seconds**, with the timer origin still marked TBD.
* Return only the four `*_hip_swing` joints to their reference angles.
* Keep the arms raised at the requested opposite `150°` deployment angles throughout.
* Use **whole-robot CoM height** with a fixed reference target.
* Require **90% ground time / at most 10% air time** over the five-second window: `4.5 s` and `0.5 s` respectively. Non-wheel ground contact remains immediate failure.
* Remove the rolling-penalty branch. Keep the existing exponential recovery-speed branch with its parameters pending.
* Permit brief high torque; use **continuous**, not accumulated, time for the five-second saturation penalty.

Three points still need clarification:


1. **Contact timing:** must all four wheels be down together for `4.5 s`, or must each wheel separately be down for `4.5 s`? I suggest the simultaneous version for your four-wheel-support goal. The per-wheel version allows staggered lifts and less time with all four down together.
2. **Recovery timing:** does the five-second window start at push onset or push end, and does missing the deadline cause a penalty or reset? I suggest push end for a finite-duration push. Also specify what happens if another push arrives before recovery finishes.
3. **Arms:** your existing pose uses `arm_deploy = -150°` and `left_arm_deploy = +150°`. Confirm that side/sign mapping; the other four arm-joint targets are still unspecified by this request.

The diagram marks remaining formulas, tolerances, contact-force thresholds, reward weights, and torque reset/rearm rules as TBD. No simulation code was changed.


---

Q: 1. use your suggest


2. penalty, later recovery,larger penalty
3. just use the p1_constant given one, that is correct

A:

Recorded your choices:

* **Support:** all four wheels must be down simultaneously for at least `4.5 s` of the five-second window. The total time without all-four support may be at most `0.5 s`.
* **Late recovery:** missing the five-second deadline causes a penalty, with a larger penalty for later recovery. It does not cause an immediate reset by itself. Non-wheel ground contact still causes failure and reset.
* **Arms:** use the configured initial pose from `p1_constants.py`, including `arm_deploy = -150°` and `left_arm_deploy = +150°`.

I replaced too much of your drawing in the previous edit. Restored your original layout, branches, node positions, and black-and-white style in [reward_design.drawio](reward_design.drawio), then applied the agreed label changes and a few small detail boxes beside the existing branches. The [preview](reward_design.preview.png) now reflects this restored structure. Future changes will build on your drawing.

For the late penalty, two options fit your description: a cost that grows while recovery is overdue, or a larger one-time cost when recovery finally happens. I suggest the growing overdue cost; a cost charged only on recovery could encourage avoiding recovery altogether. The formula and weight remain undecided, as does whether the recovery timer starts at push onset or push end.


---

Q: use a cost that grows while recovery is overdue

A:

Recorded: **after the five-second recovery deadline, apply a cost that increases while the robot remains unrecovered.** Stop that overdue cost when recovery is achieved; a missed deadline alone does not reset the episode. Non-wheel ground contact still causes failure and reset.

For the cost shape, linear growth is simpler and easier to tune; quadratic growth penalizes long delays more strongly but can dominate the other rewards. I suggest starting with linear growth when we write the formula. This is a suggestion, not an implemented formula.

About the many TBD labels: I mixed your design overview with details needed for implementation. Your main design decisions are already clear. I removed those labels from [reward_design.drawio](reward_design.drawio) and kept your layout, positions, styles, and branches. The late-recovery box now states the growing overdue cost directly.

I also read your latest edits in the diagram and retained them:

* **Support:** all four wheels down simultaneously for at least `4 s` of the five-second window (`80%`); at most `1 s` without all-four support (`20%`). This supersedes the earlier `90% / 10%` discussion.
* **Soft torque cost:** the `90%` torque-limit condition uses accumulated time exceeding `1 s`. This is separate from the large penalty for `5 s` of continuous saturation.

Before implementation, the notes still need the timer origin, recovery/contact criteria, fixed CoM target procedure, formulas/weights, and timer reset rules. Those details can be specified with each calculation instead of filling the overview diagram with placeholders. No numerical defaults were invented and no simulation code was changed.


---

Q: Just finish the reward design check which components need change in terms of calculating

A:

Checked the formulas now in your diagram. The main task choices are settled; the calculation changes below are recommendations for your Implementation mode notes. I have not changed your formulas in the diagram or written Python code.

Use **rewards** for positive contributions and **costs** for nonnegative quantities that are subtracted. Otherwise a positive “penalty” can accidentally reward the behavior.

| Component | Finding and suggested calculation change |
|----|----|
| Uprightness: `1 - (z - 9.81)^2` | With your downward-gravity convention, upright `z = -9.81` gives `-383.9444`, while inverted `z = +9.81` gives `1`. The sign is reversed. One simple alternative is `r_up = (1 - g_z) / 2`, where `g_z` is the **unit gravity direction in the IMU frame**: upright gives `1`, sideways `0.5`, inverted `0`. This uses orientation-derived gravity, not raw accelerometer output. |
| Wheel support: `1 - (t_air - 1)^2` | At air times `0`, `0.5`, and `1 s`, rewards are `0`, `0.75`, and `1`: it encourages using the entire air allowance. Instead, penalize only excess air time: `C_air = max(0, (t_air - 1 s) / 1 s)^2`. Thus `0` through `1 s` costs zero; `2 s` costs `1`. Measure `t_air` as time when **at least one wheel lacks support**, not the sum of four wheel air times. Evaluate the 80% requirement over the specified five-second window and check final all-four support separately. |
| Late recovery: `(t - 5 - 1)^2` for `t >= 5` | This is `(t - 6)^2`: the cost falls from `1` at `5 s` to `0` at `6 s`, then rises. For your chosen increasing overdue cost, use `C_late = max(0, (t - 5 s) / 1 s)^2` while unrecovered, and zero once recovered. At `5`, `6`, and `7 s`, costs are `0`, `1`, and `4`. This is a cost applied while overdue, not only when recovery occurs. |
| Continuous saturation: `exp(t) - 1`, clamped to `1` | If `t` is total continuous saturation time in seconds, the cost reaches its cap at `ln(2) = 0.693 s`; after a five-second gate it is always `1`. That is acceptable for a capped constant penalty, but does not increase with further saturation. Keep a per-motor continuous timer and an explicit five-second gate. If you want growth after five seconds, apply the exponential to **excess time after the threshold**, with a specified time scale and cap. |
| Soft overload: `log(A + 1) - 0.3` | Assuming natural logarithm and seconds, this is negative at `A = 0` and already `0.3931` at `A = 1`. Without a gate it rewards low exposure; with a one-second gate it jumps. A continuous alternative is `C_soft = ln(1 + max(0, (A - 1 s) / 1 s))`: costs at `A = 0, 1, 2 s` are `0, 0, 0.6931`. Accumulate exposure **per motor** above the chosen 90% limit. Keep this accumulated timer separate from the continuous saturation timer. |
| Hip return: `reward +1` | Define what earns it. A constant `+1` gives no pose feedback. A smooth option is `r_hip = exp(-mean(((q_i - q_ref_i) / sigma_i)^2))` over only the four `hip_swing` joints. The `sigma_i` values set the permitted error scale. An alternative is a `+1` bonus only inside explicit pose tolerances, but that gives less guidance outside them. |
| CoM height / leg adjustment: `reward +1` | Use the whole-robot CoM error, for example `r_height = exp(-((h - h_ref) / sigma_h)^2)`, with a fixed standing reference. Do not add another constant reward for “using the leg linkage”: height alone cannot identify which mechanism caused recovery. Arm holding and allowed joint motion enforce that part of the design. |
| Recovery speed: `exp(-x)` | Define `x = T_recover / tau_recover`, so the exponent has no units. If this is a recovery bonus, pay it once when recovery is confirmed. Paying `exp(-elapsed_time)` regardless of recovery rewards simply waiting. Keep the overdue cost separate. |
| Failure: `reward -1` | Combine non-wheel contacts, including arms, into one failure event. Do not charge three separate `-1` penalties for “arm contact,” “non-wheel contact,” and “fell over” when they describe the same event. Keep episode timeout separate. |

For the **recovery calculation**, combine the chosen hip, height, and attitude tolerances with final four-wheel support and a settling check. Use that result for the recovery bonus and overdue-cost switch. Otherwise touching the target briefly while moving can count as recovery. The support percentage still uses the full five-second window; do not require an early recovery bonus to wait five seconds merely to finish that measurement.

The time-dependent terms need shared per-environment state: recovery elapsed time, all-four-support time, per-motor overload exposure, per-motor continuous saturation time, and recovery-event status. Update this state once per simulation/control interval, not separately inside each reward function. Reset it at episode boundaries and specify how a new push starts a new recovery trial.

The arms stay at the configured pose through action/controller settings; the critic may observe CoM, contacts, and timers. Neither arm holding nor contact enforcement is performed by the critic itself.

Finally, separate raw calculations from final weighting. mjlab 1.6.0 normally multiplies every weighted term by the control-step duration. A desired **one-time** `-1` failure charge is therefore different from a raw term returning `-1` for one step; account for that explicitly and avoid multiplying time-based costs by `dt` twice. [Reward manager source](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/managers/reward_manager.py).

This completes the calculation review. Before coding, record the selected corrections, scales/weights, recovery tolerances, timer origin, and exposure reset/gating rules with numerical expected outputs in Implementation mode. The numerical examples above are proposed checks, not adopted implementation requirements.


---

Q: can u check for my mdp design @p1_velocity/src/p1_velocity/specification/mdp.md  and reward functon design @p1_velocity/src/p1_velocity/specification/reward_design.drawio and give me suggestion just follow up in mdp.md last quesiton  Just finish the reward design check which components need change in terms of calculating
for this project im doing the anti interference for my robot you can check around first

A:

I checked the diagram against the project files (`env_cfgs.py`, `p1_constants.py`, `p1_rl.xml`, `motors.txt`, `env_cfg.md`) and the mjlab 1.6.0 source. My previous formula corrections still apply, except the uprightness formula, which I revise below. This answer covers what the project itself changes in the calculations. No code or diagram was changed.

**Measured from the model.** I compiled `p1_rl.xml` with MuJoCo, applied the same base rotation as `get_spec()`, and set the reset pose. This is a rigid geometry check with frozen joints, not a dynamic simulation:

* Total mass is `22.688 kg`. The whole-robot CoM is about `0.3359 m` high when the front wheels touch the ground. The settled height will be a little lower (soft contact, joint sag), so measure it in simulation before fixing `h_ref`.
* The lowest arm parts sit near the arm mount on each rear lower leg (`arm_link_collision_side_23_0` and its left mirror), only about `17.6 mm` above the ground. The `±150°` deploy angle does not lift them.
* **About** `8.4°` of body roll to either side puts an arm on the ground. Pitching up to `40°` either way causes no non-wheel contact.

Under your failure rule, a sideways push that rolls the body more than about 8° is a failure unless the legs compensate. This sets the scale for the uprightness term and for push strength.

| Component | Calculation change | Reason from this project |
|----|----|----|
| Uprightness | Replace my earlier `(1 - g_z) / 2`. From 0° to 8.4° it only falls from `1` to `0.9946`, so it gives almost no warning before failure. Use the tilt angle `θ = acos(-g_z)` with a kernel sized to this margin, e.g. `exp(-(θ/σ)^2)`. Example `σ = 4°`: `1` at 0°, `0.368` at 4°, `0.012` at 8.4°, `0` inverted. | `projected_gravity_b` is taken from `base_link`; the IMU site has the same axes, so it is the IMU-frame gravity direction. |
| Non-wheel contact | Match every `.*_collision.*` geom except the four `{leg}_wheel_collision` tires. Decide whether the `*_wheel_collision_hub` geoms count as wheel; I suggest non-wheel. They sit about `16 mm` up and reach the ground only at very large tilt. | Current pattern `.*_collision\d*$` misses all arm, hand, body and shank geoms. The exclude names `LF_foot`… do not exist, so the four tires are **included**: normal wheel contact would trigger `illegal_contact`. |
| Wheel support | One contact sensor on the four tires with `reduce="netforce"`. A wheel supports when its upward force exceeds a threshold you choose (average static load is about `55.6 N` per wheel). Accumulate "not all four supported" time in your own timer. | Current wheel primaries `LF_foot`… match no geoms. The built-in `current_air_time` is per wheel and uses contact only, not force. |
| CoM height | Add a MuJoCo `subtreecom` sensor on `base_link` in `get_spec()` (like the existing `root_angmom`), or mass-weight `body_com_pos_w`. On flat ground, height = CoM z. | mjlab's `root_com_pos_w` is only `base_link`'s own CoM (about `0.466 m`). The startup `base_com` randomization (±3 cm in z) moves the whole-robot CoM by up to ±3.5 mm; keep `σ_h` above that or reduce it. |
| Hip return | Reference values: front `0°`, right rear `+18.0712°`, left rear `-18.0712°`, read from the configured default joint positions. | The inherited `pose` reward uses `HAA/HFE/KFE` names that match no P1 joint. |
| Arm hold | Remove the arms from the policy action (fixed targets) or add an arm-pose cost. | The inherited `joint_pos` action drives every actuator, arms included, so the arm pose is not guaranteed. |
| Torque costs | Use `actuator_force` (the clamped output). 90% thresholds: legs `87.3`, arms `36`, wheels `4.05 N·m`. "At limit" means \` | τ |
| Push and timer origin | Use a push with a known end, spaced wider than the 5 s window. Options: `apply_body_impulse` (finite force; its end is visible when `body_external_wrench` returns to zero) or a small event wrapper that records the push time. | Inherited `push_robot` changes velocity instantly every `1–3 s`, so the next push arrives before a 5 s window ends, and nothing in the state records when a push happened. |
| Episode length | Make episodes longer than the first push + 5 s + the overdue time you want to train. | `env_cfg.md` asks for 5 s episodes. Then a window that starts after a push never finishes: the overdue cost, the 80% support check and the 5 s saturation penalty cannot complete after a push. |
| One-time terms | At 50 Hz with dt scaling, weight `-1` on a one-step failure gives only `-0.02`. Use weight `-1/step_dt` (`-50`) for one-time terms (failure, recovery bonus). Cap the overdue cost, or make failure cost more than the largest remaining overdue cost. | Otherwise, once `(t - 5)^2` grows, touching the ground with an arm to end the episode becomes the cheaper choice. |
| Inherited terms | Remove factory rewards and terminations that are not in your diagram (velocity tracking, built-in `upright`, `pose`, foot terms, `out_of_terrain_bounds`). Decide whether to keep `fell_over` (70°) as a backup failure. | `make_velocity_env_cfg()` keeps them active unless removed, and they change your effective weights. |

**Timer state.** mjlab 1.6.0 supports class-based reward terms with `reset(env_ids)`; its own `upright` term is one. Each step runs terminations → rewards → push events → resets → observations. So rewards first see a push on the next step, and the last reward of an episode sees the timers before they reset. Also specify what the time-based terms do before the first push, when no window is active.

**Suggested shared calculations for** `mdp.py` (its header says it holds the calculation criteria, not the rewards themselves). Each can be its own Implementation mode note with numerical tests:


1. Tilt angle `θ` from projected gravity.
2. Whole-robot CoM height.
3. Per-wheel support flags and the non-wheel contact flag.
4. Per-motor torque ratio `|τ| / cap`.
5. Recovery tracker: push-end time, all-four-support time in the window, recovered flag, per-motor overload exposure, and continuous saturation time.

I suggest starting with 1 and 2. They are stateless, and the numbers above give direct test values. Settle the contact names and the episode length before the tracker (5), since its timing depends on them.


---

Q:


---

# Implementation mode

User: destination -> mdp.py, implement below according to @reward_design.drawio
```python
# Implement functions from left to right: how rewards are calculated

# this function calculates penalty when joint torque approach the limit (90%)
# input current torque output, output the  log(max(0, accumulated_t-1)+1)

# this function calculates penalty when motor torque output at max for more than 5s
# input is torque output, output will be  e^(t-5)-1, clamp1 

# this function calculates penalties for wheels in air for long time:
# input a session that the wheel in air time, output max(0, tair-1)^2

# this function tells if using limb for support consider as failure

# this function reads from IMU on the base link to check whether inclined:
# input IMU z reading, output: 1-(z/9.81+1)^2

# this function tells if arm contact with ground consider as failure:

# this function tells if fell over consider as failure:

# this function rewards if robot can go back to its original joint value place after disturbance

# this function rewards if CoM was mantained by leg:

# this function penalize if recover too late: (t-5)^2 for t>=5

```

Agent:

"""

Implemented 3 of the 10 functions in `src/p1_velocity/mdp.py`. Tests are in `tests/test_mdp.py`; all 3 pass.

* `wheel_air_time_penalty(t_air)` = `max(0, tair-1)^2`
* `imu_upright_reward(z)` = `1-(z/9.81+1)^2`
* `late_recovery_penalty(t)` = `(t-5)^2` for `t >= 5`, and `0` before 5 s. The zero before 5 s comes from the diagram ("Recovery overdue after 5 s").

Your note gives no test values, so each test uses values computed straight from your formula:

* Air time `0, 0.5, 1, 2, 3 s` → `0, 0, 0, 1, 4`
* `z = -9.81, 0, 9.81` → `1, 0, -3`
* `t = 0, 4.9, 5, 6, 7 s` → `0, 0, 0, 1, 4`

Add your own values to the note if you want independent checks.

These are pure calculations (tensor in, tensor out). They are not connected to the environment yet. `env_cfgs.py` still needs reward terms that measure `tair`, `z` and `t` and call them.

Not implemented. Questions:

1. **90% torque cost:** Which limit applies: the actuator caps in `p1_constants.py` (legs 97, arms 40, wheels 4.5 N·m)? Which motors are included? Is `accumulated_t` kept per motor, and how are motors combined (sum or max)? When does it reset (episode start or each push)?
2. **5 s saturation:** Does "at max" mean `|τ|` equal to the cap? What is the output before 5 s? There `e^(t-5)-1` is negative, so it would act as a reward. The per-motor and reset questions from 1 also apply.
3. **Limb-support failure, arm-contact failure, fell over:** What are the inputs? Are these three the same event, any non-wheel ground contact? The diagram says "arm contact = fell over". Or does "fell over" also have a tilt limit? I suggest one function: input the non-wheel contact flags, output failure true/false.
4. **Hip return:** What is the formula? A `+1` every step gives the same reward for any pose. Is it `+1` only inside a tolerance (which one), or a smooth error term?
5. **CoM height:** The same questions: formula, `h_ref`, and tolerance.
6. **Tests:** Please add expected values for each function.

Suggestions (implemented as written):

* `imu_upright_reward` expects `z` in m/s². mjlab's `projected_gravity_b` is a unit vector, so pass `9.81 * projected_gravity_b[:, 2]`. Passing the unit value gives `0.19` when upright. Near upright the reward barely changes (`0.9999` at 8.4° roll, where an arm touches the ground), so the failure rule does most of the work there.
* `late_recovery_penalty` has no upper limit. If it grows large, ending the episode by touching the ground becomes cheaper than recovering. Cap it, or make the failure cost larger.

"""

User: 
Actually, Im thinking to remove the torque cost constains as in my simulator I never meet those problems that the joints exceed the limit, the limit I was once to have is when I making the robot to stand up. To answer your questions, I think mainly those hip motors which are dm6248p and accumulated of course for each motor, I think usemax, reset at episode start?
I mean the just the limit for the actuator, before if settle actually should have reward, previouly I use the quardratic function intention is just like this if we can finish early we get reward and if late then penlaties for fucntion like -x^2+1 just did this  
Input is non tire contact with ground? 
hip return was actually part of go back to original pose, I just wanna in case it learned like using kneel to rasie the CoM to do reward hacking
you can write some tests and comment result I can validate
Finally I think we can actually remove the joint limit, since for this case may be just unnecesaary 

Agent:

"""

Updated `src/p1_velocity/mdp.py`; 6 functions now, all 6 tests in `tests/test_mdp.py` pass. Each test case has a comment with its input and expected result for you to validate.

* **Torque cost and 5 s saturation:** removed, as you decided; not implemented. The diagram still shows the torque branch; remove it there if you want the two to match.
* **Failure (limb support, arm contact, fell over):** one function, `non_tire_contact_failure(found)`. Input: non-tire ground contact counts per geom (mjlab `ContactData.found`). Output: True if any non-tire geom touches the ground. mjlab's built-in `illegal_contact` termination does the same check.
* **Hip return:** `hip_return_reward(q, q_ref, std)` = `exp(-mean(((q - q_ref)/std)^2))`. It is `1` at the reference pose, matching your `+1`.
* **CoM height:** `com_height_reward(h, h_ref, std)` = `exp(-((h - h_ref)/std)^2)`. Also `1` at the reference height.

The hip and CoM formulas are my proposal because your note has none. Their docstrings say so. If the test results look wrong, give me your formula and I'll change them. The tests use example values (`std = 0.1 rad` for hips; `h_ref = 0.336 m`, `std = 0.02 m` for CoM). The real values are set later in `env_cfgs.py`.

Suggestions:

* **Kneeling:** a knee on the ground already ends the episode through the failure rule. A strange thigh/knee pose without contact is not covered by the 4 hip joints. If that worries you, pass the thigh and knee joints to `hip_return_reward` too; it works for any joint list.
* **"Before 5 s, if settled, should have reward":** that is your recovery-speed box `e^{-x}`, which is not in this note's function list. `late_recovery_penalty` stays penalty-only, as written. To get the early reward, add recovery speed to the note with `x` defined, for example `x = T_recover / τ`.

"""

Agent:

"""

Your decisions (from chat, 2026-11-05):

1. **Late recovery: remove it.** How late doesn't matter; the robot just must not flip over. Getting back to the standing angles is already rewarded, and staying up keeps earning reward.
2. **Tires:** keep them on the ground. The 1 s allowance was there because you expected wheels to lift during a push. Use a short grace instead: 0.1 s, weight −1 per tire per second.
3. **Arm:** it may touch the ground, with a smaller penalty than falling, and as little as possible. Penalty from the first touch: −5 per second. A touch longer than 0.5 s counts as failure. The arm mounts count as arm.

Updated `src/p1_velocity/mdp.py`. Tests are in `tests/test_mdp.py` (9 tests), and each case has a comment with its input and expected result for you to validate. **Not run yet:** this PC has no project venv, so they run on the vast.ai machine first.

* **Removed:** `late_recovery_penalty` and `wheel_air_time_penalty`.
* `tire_lift_penalty(air_time, grace_s)`: the number of tires off the ground for longer than `grace_s`. With weight −1, each tire costs 1 per second off the ground, at all times (not only after a push). Examples: air times `[0.5, 2.0, 0.11, 0.05]` s → `3`; exactly `0.1` s → not counted.
* `arm_ground_contact(found)`: `1` while any arm geom touches the ground. With weight −5, contact costs 5 per second from the first touch.
* `update_contact_time(t, touching, dt)` and `arm_contact_too_long(t, tolerance_s)`: continuous arm contact longer than 0.5 s → failure. The timer goes back to 0 when the arm leaves the ground.

What each event costs with these weights:

| Event | Cost |
|----|----|
| 0.1 s arm brush | −0.5 |
| 0.5 s arm touch | −2.5 |
| One tire up for 1 s | −1 |
| Falling | about −6.5 |

Falling is −1 for the failure plus about 5.5 of lost reward. The policy looks about 2 s ahead (discount 0.99 at 50 Hz).

Diagram ([reward_design.drawio](reward_design.drawio), preview re-exported):

* Late recovery is marked "Ignored" with a red X.
* The tire box and arm box have the new rules.
* The non-wheel contact box now reads body / leg / hub.
* The joint-return box now says all 12 leg joints, which the code has used since your earlier answer.

"""

User 


