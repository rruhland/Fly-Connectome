# Returning to native Pong after numeric table-tennis practice

Returning to Pong with the actual table-trained numeric state gave different results on two independent receiving courses. The state from the 14-point qualification match met every paired noninferiority margin. The state from the 48-point match reduced later returns beyond the allowed uncertainty margin. The native source missed the first-return capability requirement, leaving registered retention unestablished for both courses.

## Paired evaluation

The source is the final score-trained numerical Pong checkpoint and the published visual role model. The receiver keeps the checkpoint’s portable position and motion gains. Its 3D lookup omits the fitted signed 2D offset because that offset belongs to the source action axis. Table practice updates the outcome totals and counts behind the numeric option selector and refits the role weights from its own camera and reward evidence. The receipt carries those actual updated states back to Pong. Pong identifies its own action mapping anew from camera observations.

Native Pong was evaluated before table practice and after each of two prespecified courses that independently started from the Pong checkpoint. The first is match zero of the 10-match, 14-point qualification at seed `15537921`. The second is one 48-point match at seed `17041069` using the same installation. Both use standard table serves, one stable camera pose per match and visual flight forecasting. The returned state is the state from the end of each course. For each option, the increases in its stored visit count and score total were checked against every receiving point’s chosen option and public reward.

Each native evaluation uses the same 36 starting matches with five points per match, the same remapped presentations and action IDs, and the same deterministic opponent at speed 0.35. Pong evaluation freezes the carried role and outcome statistics. The original criteria require a capable before-Pong player, including at least 75% first returns, 65% later returns and 15% earned points. They also require that the lower ends of the paired 95% intervals exclude declines larger than 10, 15 and 10 percentage points respectively. The bootstrap resamples complete matched native matches.

## Before table practice

The source returned 115 of 160 first approaches (71.9%), returned 115 of 159 later approaches (72.3%), and earned 38 of 180 points after a return (21.1%). The first-return rate was below the 75% capability criterion. That pre-table result prevents the two paired comparisons from establishing retention under the registered criteria.

The largest shortfall came on the first point of each native match, when the controller had to identify the remapped actions. It took a median 22 camera frames (0.73 seconds) to establish an action map. The player returned 4 of 18 opening balls that arrived within 35 native ticks (0.29 seconds) of calibration and 12 of 15 openings with more time. It returned 99 of 127 first approaches on points two through five, with the native action map already available. These are descriptive comparisons by available steering time.

## First qualification match

The scheduled first qualification match landed legal returns on 10 of 14 points and won three. Carrying its final state into Pong produced 126/160 first returns, 177/206 later returns and 39/180 earned points. The paired changes were +6.9 percentage points in first returns (95% interval +1.8 to +12.3), +13.6 points in later returns (+5.3 to +22.1), and +0.6 points in earned points (−3.9 to +5.0). All three paired margins passed.

This receiving state selected the second of the source’s two tied options on all 180 native points. With those updated outcome totals and the original role weights, first returns stayed at 126/160 and later returns reached 180/207. With the original outcome totals and the actual receiving role weights, they were 115/160 and 114/158, close to the original 115/160 and 115/159. The updated option preference accounted for the bulk of the return-rate change on this schedule.

## Extended table match

The 48-point table course landed legal returns on 36 points, produced 37 total legal returns and won three points. After carrying its exact ending state back to Pong, the player returned 111/160 first approaches and 80/134 later approaches. It earned 38/180 points after a return. Relative to the same native starts before practice, first returns fell 2.5 percentage points (paired 95% interval −8.1 to +3.1), later returns fell 12.6 points (−19.9 to −6.0), and earned points were unchanged (−5.6 to +5.6). The later-return interval exceeded the allowed 15-point decline.

The updated outcome totals changed the native option choice. The source had two tied options and selected them on 102 and 78 of its 180 native points. The state after the extended match selected the first option on all 180 points. Keeping those new outcome totals with the old role weights produced a later-return rate of 59.3% (80/135). Keeping the source totals with the actual updated role weights produced 72.2% (114/158), close to the 72.3% source rate. These controls isolate the decline to the transferred option preference on this schedule.

## Reproduce

The compressed receipt contains dedicated source and returned states, both table courses, native point histories, the crossed controls, and paired intervals. The source snapshot preserves the exact implementations used to run both continuations. The audit checked the archive hashes, recomputed the paired statistics and matched the 14-point course to the saved qualification. [Numeric retention receipt](../../artifacts/tt3/numeric-retention.json.gz) · [Source snapshot](../../artifacts/tt3/numeric-retention-code.zip) · [Receiving qualification](../../artifacts/tt3/numeric-qualification.json.gz)

Run from the repository root with the project dependencies installed. The following commands use an isolated copy of the source.

```sh
repo_dir=$PWD
run_dir=$(mktemp -d)
cp -a "$repo_dir/src" "$run_dir/src"
unzip -q "$repo_dir/artifacts/tt3/numeric-retention-code.zip" -d "$run_dir"
cd "$run_dir"
export PYTHONPATH=.:src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -c 'import gzip,json,sys; print(json.dumps(json.load(gzip.open(sys.argv[1], "rt"))["manifest"]["source"]["memory"]))' "$repo_dir/artifacts/tt3/numeric-retention.json.gz" > numeric-state.json
python -m scripts.experimental.tt3_numeric_retention prepare "$repo_dir/artifacts/tt3/pong-source.json" numeric-state.json "$repo_dir/artifacts/tt3/pong-source-receipt.json.gz" work
python -m scripts.experimental.tt3_numeric_retention finish work "$repo_dir/artifacts/tt3/numeric-qualification.json.gz" numeric-retention.json.gz
```
